from typing import cast

import numpy as np
import PIL.Image
from loguru import logger
from numpy.typing import NDArray

from osam import types

from ..yoloworld.clip import tokenize


class Sam3(types.Model):
    name = "sam3:latest"

    _blobs = {
        "image_encoder": types.Blob(
            url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.3.0/resolve/main/sam3_image_encoder.onnx",
            hash="sha256:c6f3769e9d42573806c34b663d6100b3827a4c0ecba6f45dbf39b8f3906be725",
            attachments=[
                types.Blob(
                    url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.3.0/resolve/main/sam3_image_encoder.onnx.data",
                    hash="sha256:03bd50b0703e2b04e2193ca831b7f9d5ecf40bc5287cc59b1970f56ab800c995",
                ),
            ],
        ),
        "language_encoder": types.Blob(
            url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.3.0/resolve/main/sam3_language_encoder.onnx",
            hash="sha256:b3b465935c9bf4cf5efd950589741f3da5eec9e9bfe459576989cbe565331b53",
            attachments=[
                types.Blob(
                    url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.3.0/resolve/main/sam3_language_encoder.onnx.data",
                    hash="sha256:1b03dabc657c1f2887d1b8ce2a3537467d4c033ea1f8c8be141fbca55e4b95f7",
                ),
            ],
        ),
        "decoder": types.Blob(
            url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.3.0/resolve/main/sam3_decoder.onnx",
            hash="sha256:bbc216dbf3de4742f692c2a0a0fd215ea8957956002a05f28150040a88b5dc1a",
        ),
    }

    def encode_image(self, image: NDArray[np.uint8]) -> types.ImageEmbedding:
        logger.debug("Encoding image with shape: {}", image.shape)
        original_height: int = image.shape[0]
        original_width: int = image.shape[1]

        pil_image: PIL.Image.Image = PIL.Image.fromarray(image).resize((1008, 1008))
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

        return types.ImageEmbedding(
            original_height=original_height,
            original_width=original_width,
            embedding=backbone_fpn_0,
            extra_features=[backbone_fpn_1, backbone_fpn_2, vision_pos_enc_2],
        )

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
            prompt = types.Prompt(texts=["visual"])
            logger.warning(
                "Prompt is not given, so using 'visual' as prompt: {!r}", prompt
            )
        else:
            prompt = request.prompt

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

        box_coords: NDArray[np.float32]
        box_labels: NDArray[np.int64]
        box_masks: NDArray[np.bool_]
        if prompt.points is None and prompt.point_labels is None:
            box_coords = np.array([[[0, 0, 0, 0]]], dtype=np.float32)
            box_labels = np.array([[1]], dtype=np.int64)
            box_masks = np.array([[True]], dtype=np.bool_)  # masked out
        elif prompt.points is None or prompt.point_labels is None:
            raise ValueError(
                "both points and point_labels must be provided together: "
                f"{prompt.points=}, {prompt.point_labels=}"
            )
        else:
            if len(prompt.points) != 2 or prompt.points.shape[1] != 2:
                raise ValueError(
                    "only two point prompts (left-top, right-bottom) are supported: "
                    f"{prompt.points=}, {prompt.point_labels=}"
                )
            if prompt.point_labels.tolist() != [2, 3]:
                raise ValueError(
                    "only point labels for box prompts (2: left-top, 3: right-bottom) "
                    f"are supported: {prompt.point_labels=}"
                )
            points_normalized: NDArray[np.float32] = (
                prompt.points / [original_width, original_height]
            ).astype(np.float32)
            # (x_center, y_center, width, height)
            box_coords = np.r_[
                (points_normalized[0] + points_normalized[1]) / 2,
                points_normalized[1] - points_normalized[0],
            ][None, None]
            box_labels = np.array([[1]], dtype=np.int64)
            box_masks = np.array([[False]], dtype=np.bool_)

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

        annotations: list[types.Annotation] = []
        for i in range(len(scores)):
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
