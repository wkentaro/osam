from typing import cast

import imgviz
import numpy as np
import PIL.Image
from loguru import logger
from numpy.typing import NDArray

from osam import types

from ..yoloworld.clip import tokenize


class Sam3(types.Model):
    name = "sam3:latest"
    metadata = types.ModelMetadata(
        license_name="SAM License",
        license_url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.5.0/blob/4160188c62b17bcdc8512e0fa2b6d2b53608af3d/LICENSE",
        source_url="https://github.com/wkentaro/sam3-onnx/tree/e61ac895c1a2db25c60e5a590460cbe4713e446f",
    )

    # The decoders take longer to compile for CoreML than it saves them.
    _coreml_blobs = frozenset({"image_encoder"})

    _blobs = {
        "image_encoder": types.Blob(
            url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.5.0/resolve/6ce19ce4483dd101b68a1f37b466b4e37ff2a1b2/sam3_image_encoder.onnx",
            hash="sha256:348e15d031626656c992fd2cb26a3b2a3fe03e6f038ac62b997e3e24827748bb",
        ),
        "language_encoder": types.Blob(
            url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.5.0/resolve/6ce19ce4483dd101b68a1f37b466b4e37ff2a1b2/sam3_language_encoder.onnx",
            hash="sha256:41236a2adaa762d7403781fb549b9215086c0ee3b6ee49904ef3c2955043c169",
        ),
        "decoder": types.Blob(
            url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.5.0/resolve/6ce19ce4483dd101b68a1f37b466b4e37ff2a1b2/sam3_decoder.onnx",
            hash="sha256:83c08119ec82aba0277ac8fbf5d08f7300ad7421934da5f0809e53ab74891eee",
        ),
        "point_decoder": types.Blob(
            url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.5.0/resolve/6ce19ce4483dd101b68a1f37b466b4e37ff2a1b2/sam3_point_decoder.onnx",
            hash="sha256:bf85d055c0e3fd29bf536fd750945664004b231f39e30f2d83dea3c2244ffcd5",
        ),
    }

    def encode_image(self, image: NDArray[np.uint8]) -> types.ImageEmbedding:
        logger.debug("Encoding image with shape: {}", image.shape)
        original_height: int = image.shape[0]
        original_width: int = image.shape[1]

        # SAM3 resizes with bilinear interpolation, not PIL's default bicubic.
        pil_image: PIL.Image.Image = PIL.Image.fromarray(image).resize(
            (1008, 1008), resample=PIL.Image.Resampling.BILINEAR
        )
        image_input: NDArray[np.uint8] = np.asarray(pil_image).transpose(
            2, 0, 1
        )  # HWC → CHW

        outputs = self._inference_sessions["image_encoder"].run(
            None, {"image": image_input}
        )

        vision_pos_enc_2: NDArray[np.float32] = cast(
            NDArray[np.float32], outputs[2]
        ).squeeze(axis=0)
        backbone_fpn_0: NDArray[np.float32] = cast(
            NDArray[np.float32], outputs[3]
        ).squeeze(axis=0)
        backbone_fpn_1: NDArray[np.float32] = cast(
            NDArray[np.float32], outputs[4]
        ).squeeze(axis=0)
        backbone_fpn_2: NDArray[np.float32] = cast(
            NDArray[np.float32], outputs[5]
        ).squeeze(axis=0)

        # image_embeddings, high_res_features1, high_res_features2 for points
        point_features: list[NDArray[np.float32]] = [
            cast(NDArray[np.float32], output).squeeze(axis=0) for output in outputs[6:9]
        ]

        return types.ImageEmbedding(
            original_height=original_height,
            original_width=original_width,
            embedding=backbone_fpn_0,
            extra_features=[
                backbone_fpn_1,
                backbone_fpn_2,
                vision_pos_enc_2,
                *point_features,
            ],
        )

    def default_prompt(self, height: int, width: int) -> types.Prompt:
        return types.Prompt(texts=["visual"])

    def generate(self, request: types.GenerateRequest) -> types.GenerateResponse:
        image_embedding: types.ImageEmbedding
        if request.image_embedding is not None:
            image_embedding = request.image_embedding
        elif request.image is not None:
            image_embedding = self.encode_image(image=request.image)
        else:
            raise ValueError("either image or image_embedding is required for Sam3")

        original_height: int = image_embedding.original_height
        original_width: int = image_embedding.original_width

        backbone_fpn_0: NDArray[np.float32] = image_embedding.embedding[None]
        backbone_fpn_1: NDArray[np.float32] = image_embedding.extra_features[0][None]
        backbone_fpn_2: NDArray[np.float32] = image_embedding.extra_features[1][None]
        vision_pos_enc_2: NDArray[np.float32] = image_embedding.extra_features[2][None]

        prompt: types.Prompt
        if request.prompt is None:
            prompt = self.default_prompt(height=original_height, width=original_width)
            logger.warning(
                "Prompt is not given, so using 'visual' as prompt: {!r}", prompt
            )
        else:
            prompt = request.prompt

        if (
            prompt.points is not None
            and prompt.point_labels is not None
            and len(prompt.point_labels) > 0
            and np.isin(prompt.point_labels, [0, 1]).all()
        ):
            if prompt.texts:
                raise ValueError("point prompts can't be combined with text")
            return self._generate_from_points(
                image_embedding=image_embedding,
                points=prompt.points,
                point_labels=prompt.point_labels,
            )

        text_prompt: str = "visual"
        if prompt.texts:
            if len(prompt.texts) > 1:
                logger.warning(
                    "Only first text prompt is used, ignoring: {}", prompt.texts[1:]
                )
            text_prompt = prompt.texts[0]

        score_threshold: float = (
            0 if prompt.score_threshold is None else prompt.score_threshold
        )

        tokens: NDArray[np.int64] = tokenize(texts=[text_prompt], context_length=32)
        outputs = self._inference_sessions["language_encoder"].run(
            None, {"tokens": tokens}
        )
        language_mask: NDArray[np.bool_] = cast(NDArray[np.bool_], outputs[0])
        language_features: NDArray[np.float32] = cast(NDArray[np.float32], outputs[1])

        # The decoder takes a fixed number of boxes, and masks out the unused ones.
        (box_input,) = [
            i
            for i in self._inference_sessions["decoder"].get_inputs()
            if i.name == "box_coords"
        ]
        max_boxes: int = box_input.shape[0]
        box_coords: NDArray[np.float32] = np.zeros((max_boxes, 1, 4), np.float32)
        box_labels: NDArray[np.int64] = np.ones((max_boxes, 1), np.int64)
        box_masks: NDArray[np.bool_] = np.ones((1, max_boxes), np.bool_)
        if prompt.points is not None and prompt.point_labels is not None:
            num_boxes: int = len(prompt.points) // 2
            pairs: list[list[int]] = (
                []
                if len(prompt.points) % 2
                else prompt.point_labels.reshape(-1, 2).tolist()
            )
            if not pairs or any(pair not in ([2, 3], [4, 5]) for pair in pairs):
                raise ValueError(
                    "points (labels 0, 1) or boxes as point label pairs "
                    "(2, 3: positive left-top, right-bottom; 4, 5: negative) "
                    "are supported, not both: "
                    f"{prompt.point_labels=}"
                )
            if num_boxes > max_boxes:
                raise ValueError(f"at most {max_boxes} boxes are supported")
            corners: NDArray[np.floating] = prompt.points.reshape(num_boxes, 2, 2) / [
                original_width,
                original_height,
            ]
            # (x_center, y_center, width, height)
            box_coords[:num_boxes, 0] = np.c_[
                corners.mean(axis=1), corners[:, 1] - corners[:, 0]
            ]
            box_labels[:num_boxes, 0] = [pair == [2, 3] for pair in pairs]
            box_masks[0, :num_boxes] = False

        outputs = self._inference_sessions["decoder"].run(
            None,
            {
                "backbone_fpn_0": backbone_fpn_0,
                "backbone_fpn_1": backbone_fpn_1,
                "backbone_fpn_2": backbone_fpn_2,
                "vision_pos_enc_2": vision_pos_enc_2,
                "language_mask": language_mask,
                "language_features": language_features,
                "box_coords": box_coords,
                "box_labels": box_labels,
                "box_masks": box_masks,
            },
        )
        boxes_norm: NDArray[np.float32] = cast(NDArray[np.float32], outputs[0])
        scores: NDArray[np.float32] = cast(NDArray[np.float32], outputs[1])
        masks_native: NDArray[np.float32] = cast(NDArray[np.float32], outputs[2])
        mask_h, mask_w = masks_native.shape[2], masks_native.shape[3]

        # Scale normalized boxes to original image coordinates
        boxes = boxes_norm * np.array(
            [original_width, original_height, original_width, original_height],
            dtype=np.float32,
        )
        # Scale normalized boxes to native mask coordinates
        boxes_mask = boxes_norm * np.array(
            [mask_w, mask_h, mask_w, mask_h], dtype=np.float32
        )

        # Keep the highest-scoring annotations, in the decoder's order.
        keep = np.sort(np.argsort(-scores, kind="stable")[: prompt.max_annotations])
        annotations: list[types.Annotation] = []
        for i in keep:
            if scores[i] < score_threshold:
                continue
            box = boxes[i]
            bb = types.BoundingBox(
                xmin=round(box[0]),
                ymin=round(box[1]),
                xmax=round(box[2]),
                ymax=round(box[3]),
            )
            bbox_w = bb.xmax - bb.xmin + 1
            bbox_h = bb.ymax - bb.ymin + 1
            # Crop in native mask space, then resize to bbox dimensions
            bm = boxes_mask[i].astype(int)
            native_crop = masks_native[i, 0, bm[1] : bm[3] + 1, bm[0] : bm[2] + 1]
            cropped_mask: NDArray[np.bool_] = (
                np.asarray(
                    PIL.Image.fromarray(native_crop).resize(
                        (bbox_w, bbox_h),
                        resample=PIL.Image.Resampling.BILINEAR,
                    )
                )
                > 0.5
            )
            annotations.append(
                types.Annotation(
                    mask=cropped_mask,
                    bounding_box=bb,
                    text=text_prompt,
                    score=float(scores[i]),
                )
            )

        return types.GenerateResponse(
            model=self.name,
            image_embedding=image_embedding,
            annotations=annotations,
        )

    def _generate_from_points(
        self,
        image_embedding: types.ImageEmbedding,
        points: NDArray[np.floating],
        point_labels: NDArray[np.integer],
    ) -> types.GenerateResponse:
        if len(image_embedding.extra_features) < 6:
            raise ValueError(
                "point prompts need an image embedding from this osam version, "
                "encode the image again"
            )
        height: int = image_embedding.original_height
        width: int = image_embedding.original_width
        inputs: dict[str, NDArray] = {
            "image_embeddings": image_embedding.extra_features[3][None],
            "high_res_features1": image_embedding.extra_features[4][None],
            "high_res_features2": image_embedding.extra_features[5][None],
            "point_coords": (points / [width, height] * 1008).astype(np.float32)[None],
            "point_labels": point_labels.astype(np.float32)[None],
            "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        }
        session = self._inference_sessions["point_decoder"]
        iou_predictions, low_res_masks = cast(
            list[NDArray[np.float32]], session.run(None, inputs)
        )
        # Refine with the best mask from the first pass as the mask prompt.
        inputs["mask_input"] = low_res_masks[:, [iou_predictions[0].argmax()]]
        inputs["has_mask_input"] = np.array([1], dtype=np.float32)
        iou_predictions, low_res_masks = cast(
            list[NDArray[np.float32]], session.run(None, inputs)
        )
        best: int = int(iou_predictions[0].argmax())

        # ponytail: no filling of small background holes, unlike SAM2's
        # postprocessing; add it if masks show pinholes.
        mask: NDArray[np.bool_] = (
            np.asarray(
                PIL.Image.fromarray(low_res_masks[0, best]).resize(
                    (width, height), resample=PIL.Image.Resampling.BILINEAR
                )
            )
            > 0
        )
        annotations: list[types.Annotation] = []
        if mask.any():
            ymin, xmin, ymax, xmax = imgviz.masks_to_bboxes(masks=[mask])[0].astype(int)
            annotations.append(
                types.Annotation(
                    mask=mask[ymin : ymax + 1, xmin : xmax + 1],
                    bounding_box=types.BoundingBox(
                        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax
                    ),
                    score=float(iou_predictions[0, best]),
                )
            )
        return types.GenerateResponse(
            model=self.name,
            image_embedding=image_embedding,
            annotations=annotations,
        )
