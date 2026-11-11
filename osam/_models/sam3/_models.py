from typing import Final
from typing import cast

import numpy as np
import onnxruntime
import PIL.Image
from loguru import logger
from numpy.typing import NDArray
from typing_extensions import override

from osam import _onnx
from osam import _types as types

from .._clip import tokenize

# sam3-onnx's models, by file name without the sam3_ prefix and .onnx.
_FILES: Final[dict[str, types.Blob]] = {
    "image_encoder": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_image_encoder.onnx",
        hash="sha256:7431ce5878b54885a59a48af08739de927793d0c9e8acc368924b2b7b1e919da",
    ),
    "image_encoder_fp16": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_image_encoder_fp16.onnx",
        hash="sha256:42db5e7c65f18c536f35e14c591fdb9ffeb6f24339add59043763afde9a7fda5",
    ),
    "image_encoder_fused_fp16": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_image_encoder_fused_fp16.onnx",
        hash="sha256:9bb34ce7774c13b36510db4a4c32b15b9b38b6f0439a0d8a4245ac1b20f32708",
    ),
    "language_encoder": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_language_encoder.onnx",
        hash="sha256:1f92c701be59c060182cc62ea5653e4c4c3cfc4a13835ea9db6d669267ecd063",
    ),
    "language_encoder_fp16": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_language_encoder_fp16.onnx",
        hash="sha256:7cec8f527d75461e3f0c2e97134d38d9f57194929b8f5a3e81f03555aaab1588",
    ),
    "decoder": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_decoder.onnx",
        hash="sha256:0a5e517a950abbe42133a8f57b9d16184a939803fe249014afb7e02472ebd20e",
    ),
    "decoder_fp16": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_decoder_fp16.onnx",
        hash="sha256:b5a24f389bdaee26257ba71946a52f629ff1369e47ee8a1f3e2aa14922bb67f3",
    ),
    "point_decoder": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_point_decoder.onnx",
        hash="sha256:b216df022f3767a1d9c3252902368a653ba9a2a14be3db7f85720dd27cabdd61",
    ),
    "point_decoder_fp16": types.Blob(
        url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/resolve/a800a982cf32ace0638ffa16b86c742ff152796b/sam3_point_decoder_fp16.onnx",
        hash="sha256:92ad10439ac5fb86d1a2cfa603e8ed34000b079d2d6eab0cef50297193fb7825",
    ),
}


_ORT_VERSION: Final = tuple(int(v) for v in onnxruntime.__version__.split(".")[:2])


def _file(*, name: str, device: str) -> str:
    # The copy of a model that runs fastest on a session's device: float16 on
    # a GPU, which runs it up to twice as fast, and float32 on the CPU, which
    # lacks float16 kernels. On CUDA and WebGPU, the image encoder's attention
    # runs as onnxruntime's MultiHeadAttention, which CoreML takes minutes to
    # compile. onnxruntime before 1.24, the last on Python 3.10, runs the
    # float16 image encoder 10 times as slow as float32 on CoreML.
    if device == "cpu" or (device == "coreml" and _ORT_VERSION < (1, 24)):
        return name
    if name == "image_encoder" and device in ("cuda", "webgpu"):
        return f"{name}_fused_fp16"
    return f"{name}_fp16"


class Sam3(types.Model):
    name = "sam3:latest"
    metadata = types.ModelMetadata(
        license_name="SAM License",
        license_url="https://huggingface.co/wkentaro/sam3-onnx-models-v0.7.0/blob/a800a982cf32ace0638ffa16b86c742ff152796b/LICENSE",
        source_url="https://github.com/wkentaro/sam3-onnx/tree/2ad7d5be6c5e53a32572026c94054ea2bee0294c",
    )

    prompt_kinds = frozenset({"points", "box", "texts", "example_boxes"})

    _blobs = _FILES

    # CoreML runs the image encoder 3 times as fast as WebGPU does. It takes
    # 11 s to compile the decoder for 140 ms less per run, and can't load the
    # point decoder, whose point_coords dimension is unbounded.
    _coreml_blobs = frozenset({"image_encoder"})

    @classmethod
    @override
    def _device_blobs(cls, *, device: str) -> dict[str, types.Blob]:
        blobs: dict[str, types.Blob] = {}
        for name in ("image_encoder", "language_encoder", "decoder", "point_decoder"):
            session: str = _onnx.session_device(
                device=device, coreml=name in cls._coreml_blobs
            )
            blobs[name] = _FILES[_file(name=name, device=session)]
        return blobs

    @override
    def encode_image(self, *, image: NDArray[np.uint8]) -> types.ImageEmbedding:
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

        # By name, without vision_pos_enc_0 and vision_pos_enc_1: 106 MB of
        # constants that no decoder reads, copied from the GPU on every run.
        output_names: list[str] = [
            "vision_pos_enc_2",
            "backbone_fpn_0",
            "backbone_fpn_1",
            "backbone_fpn_2",
            # for points
            "image_embeddings",
            "high_res_features1",
            "high_res_features2",
        ]
        # With the batch dimension the decoders take.
        (
            vision_pos_enc_2,
            backbone_fpn_0,
            backbone_fpn_1,
            backbone_fpn_2,
            *point_features,
        ) = _onnx.run_on_device(
            session=self._inference_sessions["image_encoder"],
            names=output_names,
            inputs={"image": image_input},
        )

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

    @classmethod
    @override
    def default_prompt(cls, *, height: int, width: int) -> types.Prompt:
        return types.Prompt(texts=["visual"])

    @override
    def _decode(
        self, *, embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> list[types.Annotation]:
        original_height: int = embedding.original_height
        original_width: int = embedding.original_width

        score_threshold: float = (
            0.1 if prompt.score_threshold is None else prompt.score_threshold
        )
        max_annotations: int = (
            100 if prompt.max_annotations is None else prompt.max_annotations
        )

        if prompt.points is not None or prompt.box is not None:
            # One object, as in SAM and SAM2.
            points, point_labels = types.point_inputs(prompt=prompt)
            return self._generate_from_points(
                image_embedding=embedding,
                points=points,
                point_labels=point_labels,
                score_threshold=score_threshold,
                max_annotations=max_annotations,
            )

        # The decoder takes a fixed number of boxes, and masks out the unused ones.
        (box_input,) = (
            i
            for i in self._inference_sessions["decoder"].get_inputs()
            if i.name == "box_coords"
        )
        max_boxes: int = box_input.shape[0]
        box_coords: NDArray[np.float32] = np.zeros((max_boxes, 1, 4), np.float32)
        box_labels: NDArray[np.int64] = np.ones((max_boxes, 1), np.int64)
        box_masks: NDArray[np.bool_] = np.ones((1, max_boxes), np.bool_)
        if prompt.example_boxes:
            num_boxes: int = len(prompt.example_boxes)
            if num_boxes > max_boxes:
                raise ValueError(f"at most {max_boxes} boxes are supported")
            corners: NDArray[np.floating] = np.reshape(
                prompt.example_boxes, (num_boxes, 2, 2)
            ) / [original_width, original_height]
            # (x_center, y_center, width, height)
            box_coords[:num_boxes, 0] = np.c_[
                corners.mean(axis=1), corners[:, 1] - corners[:, 0]
            ]
            box_labels[:num_boxes, 0] = prompt.example_box_labels or [1] * num_boxes
            box_masks[0, :num_boxes] = False

        # One decoder run per text, joined so max_annotations caps them all, as
        # for YOLO-World. "visual" alone finds every object like the examples.
        texts: list[str] = prompt.texts or ["visual"]
        runs: list[list[NDArray[np.float32]]] = []
        for text in texts:
            tokens: NDArray[np.int64] = tokenize(texts=[text], context_length=32)
            language_mask, language_features = self._inference_sessions[
                "language_encoder"
            ].run(["text_attention_mask", "text_memory"], {"tokens": tokens})
            outputs = self._inference_sessions["decoder"].run(
                ["boxes", "scores", "masks"],
                {
                    "backbone_fpn_0": embedding.embedding,
                    "backbone_fpn_1": embedding.extra_features[0],
                    "backbone_fpn_2": embedding.extra_features[1],
                    "vision_pos_enc_2": embedding.extra_features[2],
                    "language_mask": language_mask,
                    "language_features": language_features,
                    "box_coords": box_coords,
                    "box_labels": box_labels,
                    "box_masks": box_masks,
                },
            )
            runs.append(cast(list[NDArray[np.float32]], outputs))
        # Boxes of objects cut off by the image edge can extend past it.
        boxes_norm: NDArray[np.float32] = np.clip(
            np.concatenate([run[0] for run in runs]), 0, 1
        )
        scores: NDArray[np.float32] = np.concatenate([run[1] for run in runs])
        masks_native: NDArray[np.float32] = np.concatenate([run[2] for run in runs])
        row_texts: list[str] = [text for text, run in zip(texts, runs) for _ in run[1]]
        mask_h, mask_w = masks_native.shape[2], masks_native.shape[3]

        # Scale normalized boxes to inclusive pixel coordinates in the image
        boxes = np.minimum(
            boxes_norm
            * np.array(
                [original_width, original_height, original_width, original_height],
                dtype=np.float32,
            ),
            [original_width - 1, original_height - 1] * 2,
        )
        # Scale normalized boxes to native mask coordinates
        boxes_mask = boxes_norm * np.array(
            [mask_w, mask_h, mask_w, mask_h], dtype=np.float32
        )

        # Keep the highest-scoring annotations, in text then decoder order.
        keep = np.sort(np.argsort(-scores, kind="stable")[:max_annotations])
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
                    text=row_texts[i],
                    score=float(scores[i]),
                )
            )

        return annotations

    def _generate_from_points(
        self,
        *,
        image_embedding: types.ImageEmbedding,
        points: NDArray[np.float32],
        point_labels: NDArray[np.float32],
        score_threshold: float,
        max_annotations: int,
    ) -> list[types.Annotation]:
        height: int = image_embedding.original_height
        width: int = image_embedding.original_width
        inputs: dict[str, types.Feature] = {
            "image_embeddings": image_embedding.extra_features[3],
            "high_res_features1": image_embedding.extra_features[4],
            "high_res_features2": image_embedding.extra_features[5],
            "point_coords": (points / [width, height] * 1008).astype(np.float32)[None],
            "point_labels": point_labels.astype(np.float32)[None],
            "mask_input": np.zeros((1, 1, 288, 288), dtype=np.float32),
            "has_mask_input": np.array([0], dtype=np.float32),
        }
        session = self._inference_sessions["point_decoder"]
        names: list[str] = ["iou_predictions", "low_res_masks"]
        iou_predictions, low_res_masks = cast(
            list[NDArray[np.float32]], session.run(names, inputs)
        )
        # Refine with the best mask from the first pass as the mask prompt.
        inputs["mask_input"] = low_res_masks[:, [iou_predictions[0].argmax()]]
        inputs["has_mask_input"] = np.array([1], dtype=np.float32)
        iou_predictions, low_res_masks = cast(
            list[NDArray[np.float32]], session.run(names, inputs)
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
        score = float(iou_predictions[0, best])
        if score < score_threshold or max_annotations == 0:
            return []
        return types.annotations_from_mask(mask=mask, score=score)
