import pathlib
from typing import Final
from typing import cast

import numpy as np
import onnxruntime
from numpy.typing import NDArray

# A single NonMaxSuppression node with no weights, so it ships with the package
# instead of being downloaded.
_NON_MAXIMUM_SUPPRESSION_ONNX: Final = (
    pathlib.Path(__file__).parent / "_data" / "non_maximum_suppression.onnx"
)

_non_maximum_suppression_inference_session: onnxruntime.InferenceSession | None = None


def non_maximum_suppression(
    *,
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating],
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
) -> tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]
]:
    """Keep the best of the overlapping boxes, per class, on CPU.

    boxes (N, 4) as xmin, ymin, xmax, ymax; scores (N, C), a score per class.
    A box can be kept for more than one class. Return the kept boxes, their
    scores, their classes and their indices into boxes, at most
    max_num_detections of them.
    """
    global _non_maximum_suppression_inference_session
    if _non_maximum_suppression_inference_session is None:
        _non_maximum_suppression_inference_session = onnxruntime.InferenceSession(
            str(_NON_MAXIMUM_SUPPRESSION_ONNX), providers=["CPUExecutionProvider"]
        )
    inference_session = _non_maximum_suppression_inference_session

    outputs = inference_session.run(
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
    )
    selected_indices = cast(NDArray[np.int64], outputs[0])
    labels = selected_indices[:, 1]
    box_indices = selected_indices[:, 2]
    boxes = boxes[box_indices]
    scores = scores[box_indices, labels]
    indices = box_indices

    if len(boxes) > max_num_detections:
        keep_indices = np.argsort(scores)[-max_num_detections:]
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]
        indices = indices[keep_indices]

    return boxes, scores, labels, indices
