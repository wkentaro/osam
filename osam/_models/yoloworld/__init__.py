from typing import Tuple

import imgviz
import numpy as np
import onnxruntime
from loguru import logger

from ... import types
from . import clip


class _YoloWorld(types.Model):
    _image_size: int

    def generate(self, request: types.GenerateRequest) -> types.GenerateResponse:
        if request.prompt is None:
            prompt = types.Prompt(texts=["anything"])
            logger.warning(
                "Prompt is not given, so using 'anything' as prompt: {prompt!r}",
                prompt=prompt,
            )
        else:
            prompt = request.prompt

        if prompt.texts is None:
            raise ValueError("prompt.texts is required: prompt=%r", prompt)
        token = clip.tokenize(texts=prompt.texts + [" "])
        (text_features,) = self._inference_sessions["textual"].run(
            None, {"input": token}
        )
        text_features = text_features / np.linalg.norm(
            text_features, ord=2, axis=1, keepdims=True
        )
        #
        if request.image is None:
            raise ValueError("request.image is required: request=%r", request)
        transformed_image, original_image_hw, padding_hw = _transform_image(
            image=request.image, image_size=self._image_size
        )
        scores, bboxes = self._inference_sessions["yolo"].run(
            output_names=["scores", "boxes"],
            input_feed={
                "images": transformed_image[None],
                "text_features": text_features[None],
            },
        )
        scores = scores[0]
        bboxes = bboxes[0]
        #
        iou_threshold = 1.0 if prompt.iou_threshold is None else prompt.iou_threshold
        score_threshold = (
            0.0 if prompt.score_threshold is None else prompt.score_threshold
        )
        max_annotations = (
            len(bboxes) if prompt.max_annotations is None else prompt.max_annotations
        )
        bboxes, scores, labels = _non_maximum_suppression(
            inference_session=self._inference_sessions["nms"],
            boxes=bboxes,
            scores=scores,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            max_num_detections=max_annotations,
        )
        bboxes = _untransform_bboxes(
            bboxes=bboxes,
            image_size=self._image_size,
            original_image_hw=original_image_hw,
            padding_hw=padding_hw,
        )
        annotations = [
            types.Annotation(
                bounding_box=types.BoundingBox(
                    xmin=bbox[0], ymin=bbox[1], xmax=bbox[2], ymax=bbox[3]
                ),
                text=prompt.texts[label],
                score=score,
            )
            for bbox, label, score in zip(bboxes, labels, scores)
        ]
        return types.GenerateResponse(
            model=self.name,
            image_embedding=None,
            annotations=annotations,
        )


class YoloWorldXL(_YoloWorld):
    name: str = "yoloworld:latest"

    _image_size = 640
    _blobs = {
        "textual": types.Blob(
            url="https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/ViT-B-32/textual.onnx",
            hash="sha256:55c85d8cbb096023781c1d13c557eb95d26034c111bd001b7360fdb7399eec68",
        ),
        "yolo": types.Blob(
            url="https://github.com/wkentaro/yolo-world-onnx/releases/download/v0.1.0/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.onnx",
            hash="sha256:92660c6456766439a2670cf19a8a258ccd3588118622a15959f39e253731c05d",
        ),
        "nms": types.Blob(
            url="https://github.com/wkentaro/yolo-world-onnx/releases/download/v0.1.0/non_maximum_suppression.onnx",
            hash="sha256:328310ba8fdd386c7ca63fc9df3963cc47b1268909647abd469e8ebdf7f3d20a",
        ),
    }


def _transform_image(
    image: np.ndarray, image_size: int
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    height, width = image.shape[:2]

    scale = image_size / max(height, width)
    image_resized = imgviz.resize(
        image,
        height=int(height * scale),
        width=int(width * scale),
        interpolation="linear",
    )
    pad_height = image_size - image_resized.shape[0]
    pad_width = image_size - image_resized.shape[1]
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
    input_image = image_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return input_image, (height, width), (pad_height, pad_width)


def _untransform_bboxes(
    bboxes: np.ndarray,
    image_size: int,
    original_image_hw: Tuple[int, int],
    padding_hw: Tuple[int, int],
) -> np.ndarray:
    bboxes -= np.array([padding_hw[1] // 2, padding_hw[0] // 2] * 2)
    bboxes /= image_size / max(original_image_hw)
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, original_image_hw[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, original_image_hw[0])
    bboxes = bboxes.round().astype(int)
    return bboxes


def _non_maximum_suppression(
    inference_session: onnxruntime.InferenceSession,
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_indices = inference_session.run(
        output_names=["selected_indices"],
        input_feed={
            "boxes": boxes[None, :, :],
            "scores": scores[None, :, :].transpose(0, 2, 1),
            "max_output_boxes_per_class": np.array(
                [max_num_detections], dtype=np.int64
            ),
            "iou_threshold": np.array([iou_threshold], dtype=np.float32),
            "score_threshold": np.array([score_threshold], dtype=np.float32),
        },
    )[0]
    labels = selected_indices[:, 1]
    box_indices = selected_indices[:, 2]
    boxes = boxes[box_indices]
    scores = scores[box_indices, labels]

    if len(boxes) > max_num_detections:
        keep_indices = np.argsort(scores)[-max_num_detections:]
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

    return boxes, scores, labels
