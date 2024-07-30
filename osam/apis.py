from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
import onnxruntime

from . import _models
from . import types

running_model: Optional[types.Model] = None

registered_model_types: List[Type] = [
    _models.EfficientSam10m,
    _models.EfficientSam30m,
    _models.Sam100m,
    _models.Sam300m,
    _models.Sam600m,
    _models.YoloWorldXL,
]


def get_model_type_by_name(name: str) -> Type:
    model_name: str
    if ":" in name:
        model_name = name
    else:
        model_name = f"{name}:latest"

    for cls in registered_model_types:
        if cls.name == model_name:
            break
    else:
        raise ValueError(f"Model {name!r} not found.")
    return cls


def generate(request: types.GenerateRequest) -> types.GenerateResponse:
    global running_model

    model_type = get_model_type_by_name(name=request.model)
    if running_model is None or running_model.name != model_type.name:
        running_model = model_type()
    assert running_model is not None

    response: types.GenerateResponse = running_model.generate(request=request)
    return response


_non_maximum_suppression_inference_session: Optional[onnxruntime.InferenceSession] = (
    None
)


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    global _non_maximum_suppression_inference_session
    if _non_maximum_suppression_inference_session is None:
        blob = types.Blob(
            url="https://github.com/wkentaro/yolo-world-onnx/releases/download/v0.1.0/non_maximum_suppression.onnx",  # noqa
            hash="sha256:328310ba8fdd386c7ca63fc9df3963cc47b1268909647abd469e8ebdf7f3d20a",
        )
        blob.pull()
        _non_maximum_suppression_inference_session = onnxruntime.InferenceSession(
            blob.path, providers=["CPUExecutionProvider"]
        )
    inference_session = _non_maximum_suppression_inference_session

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
