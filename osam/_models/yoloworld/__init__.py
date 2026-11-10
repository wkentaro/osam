from typing import cast

import imgviz
import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from ... import _nms
from ... import _types as types
from .. import _clip as clip

__all__ = [
    "YoloWorldXL",
]


class _YoloWorld(types.Model):
    metadata = types.ModelMetadata(
        license_name="GNU General Public License v3.0",
        license_url="https://github.com/wkentaro/yolo-world-onnx/blob/959674c1f64095e78ee3d9bb246d01f2443caf35/LICENSE",
        source_url="https://github.com/wkentaro/yolo-world-onnx/blob/959674c1f64095e78ee3d9bb246d01f2443caf35/ARTIFACTS.md",
        license_spdx="GPL-3.0-only",
    )

    prompt_kinds = frozenset({"texts"})

    _input_size: int

    @classmethod
    @override
    def default_prompt(cls, *, height: int, width: int) -> types.Prompt:
        return types.Prompt(texts=["anything"])

    @override
    def encode_image(self, *, image: NDArray[np.uint8]) -> types.ImageEmbedding:
        # The network runs on the image and the texts together, so the
        # embedding is the image resized and padded to the input size.
        return types.ImageEmbedding(
            original_height=image.shape[0],
            original_width=image.shape[1],
            embedding=_transform_image(image=image, input_size=self._input_size),
        )

    @override
    def _decode(
        self, *, embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> list[types.Annotation]:
        texts: list[str] | None = prompt.texts
        assert texts is not None  # decode checks prompt_kinds
        token = clip.tokenize(texts=texts + [" "])
        outputs = self._inference_sessions["textual"].run(None, {"input": token})
        text_features: NDArray[np.float32] = cast(NDArray[np.float32], outputs[0])
        text_features = text_features / np.linalg.norm(
            text_features, ord=2, axis=1, keepdims=True
        )
        #
        yolo_outputs = self._inference_sessions["yolo"].run(
            output_names=["scores", "boxes"],
            input_feed={
                "images": embedding.embedding[None],
                "text_features": text_features[None],
            },
        )
        scores = cast(NDArray[np.float32], yolo_outputs[0])[0]
        # The last class is the " " padding text, which no one asked for.
        scores = scores[:, : len(texts)]
        bboxes = cast(NDArray[np.float32], yolo_outputs[1])[0]
        #
        iou_threshold = 0.5 if prompt.iou_threshold is None else prompt.iou_threshold
        score_threshold = (
            0.1 if prompt.score_threshold is None else prompt.score_threshold
        )
        max_annotations = (
            100 if prompt.max_annotations is None else prompt.max_annotations
        )
        bboxes, scores, labels, _indices = _nms.non_maximum_suppression(
            boxes=bboxes,
            scores=scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_num_detections=max_annotations,
        )
        bboxes = _untransform_bboxes(
            bboxes=bboxes,
            input_size=self._input_size,
            original_image_hw=(embedding.original_height, embedding.original_width),
        )
        return [
            types.Annotation(
                bounding_box=types.BoundingBox(
                    xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3]
                ),
                text=texts[label],
                score=score,
            )
            for bbox, label, score in zip(bboxes, labels, scores)
        ]


class YoloWorldXL(_YoloWorld):
    name: str = "yoloworld:latest"

    _input_size = 640
    _blobs = {
        "textual": types.Blob(
            url="https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx",
            hash="sha256:55c85d8cbb096023781c1d13c557eb95d26034c111bd001b7360fdb7399eec68",
        ),
        "yolo": types.Blob(
            url="https://github.com/wkentaro/yolo-world-onnx/releases/download/v0.1.0/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",
            hash="sha256:92660c6456766439a2670cf19a8a258ccd3588118622a15959f39e253731c05d",
        ),
    }


def _padding_hw(
    *, original_image_hw: tuple[int, int], input_size: int
) -> tuple[int, int]:
    # The image scales so its longer side fits input_size; the rest is padding.
    height, width = original_image_hw
    scale = input_size / max(height, width)
    return input_size - int(height * scale), input_size - int(width * scale)


def _transform_image(*, image: np.ndarray, input_size: int) -> NDArray[np.float32]:
    pad_height, pad_width = _padding_hw(
        original_image_hw=image.shape[:2], input_size=input_size
    )
    image_resized = imgviz.resize(
        image,
        height=input_size - pad_height,
        width=input_size - pad_width,
        interpolation="linear",
    )
    image_resized = np.pad(
        image_resized,
        (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
            (0, 0),
        ),
        mode="constant",
        constant_values=114,
    )
    return image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0


def _untransform_bboxes(
    *,
    bboxes: np.ndarray,
    input_size: int,
    original_image_hw: tuple[int, int],
) -> np.ndarray:
    pad_height, pad_width = _padding_hw(
        original_image_hw=original_image_hw, input_size=input_size
    )
    bboxes -= np.array([pad_width // 2, pad_height // 2] * 2)
    bboxes /= input_size / max(original_image_hw)
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, original_image_hw[1] - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, original_image_hw[0] - 1)
    bboxes = bboxes.round().astype(int)
    return bboxes
