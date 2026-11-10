import numpy as np
import pytest

import osam
from osam import _nms
from osam import types


def test_non_maximum_suppression_runs_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_nms, "_non_maximum_suppression_inference_session", None)

    def fail_pull(*args: object, **kwargs: object) -> None:
        raise AssertionError("The graph ships with the package")

    monkeypatch.setattr(types.Blob, "pull", fail_pull)

    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [52, 52, 62, 62]],
        dtype=np.float32,
    )
    scores = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.6]], dtype=np.float32
    )

    kept_boxes, kept_scores, labels, indices = osam.non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        score_threshold=0.25,
        max_num_detections=10,
    )

    # Pinned against the graph osam downloaded before it was packaged.
    assert indices.tolist() == [0, 2, 2, 3]
    assert labels.tolist() == [0, 0, 1, 1]
    assert kept_scores.tolist() == pytest.approx([0.9, 0.3, 0.7, 0.6])
    assert kept_boxes.tolist() == boxes[[0, 2, 2, 3]].tolist()


def test_non_maximum_suppression_keeps_its_order_when_it_trims() -> None:
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [52, 52, 62, 62]],
        dtype=np.float32,
    )
    scores = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.6]], dtype=np.float32
    )

    _, kept_scores, labels, indices = osam.non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        score_threshold=0.25,
        max_num_detections=3,
    )

    # The lowest of the four kept above, 0.3, is the one dropped.
    assert indices.tolist() == [0, 2, 3]
    assert labels.tolist() == [0, 1, 1]
    assert kept_scores.tolist() == pytest.approx([0.9, 0.7, 0.6])
