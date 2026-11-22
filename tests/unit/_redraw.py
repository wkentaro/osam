import numpy as np

from osam import types


def _lerp(*, grid: np.ndarray, t: np.ndarray, axis: int) -> np.ndarray:
    t = np.clip(t, 0, grid.shape[axis] - 1)
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, grid.shape[axis] - 1)
    w1 = np.expand_dims(t - i0, 1 - axis)
    return np.take(grid, i0, axis) * (1 - w1) + np.take(grid, i1, axis) * w1


def redrawn(*, annotation: types.Annotation) -> np.ndarray:
    assert annotation.bounding_box is not None
    assert annotation.mask_logits is not None
    assert annotation.mask_logits_box is not None
    bb = annotation.bounding_box
    xmin, ymin, xmax, ymax = annotation.mask_logits_box
    grid = annotation.mask_logits.astype(np.float64)
    rows, cols = grid.shape
    xs = (np.arange(bb.xmin, bb.xmax + 1) + 0.5 - xmin) * cols / (xmax - xmin) - 0.5
    ys = (np.arange(bb.ymin, bb.ymax + 1) + 0.5 - ymin) * rows / (ymax - ymin) - 0.5
    return _lerp(grid=_lerp(grid=grid, t=ys, axis=0), t=xs, axis=1)


def distance_where_redraw_differs(*, annotation: types.Annotation) -> float:
    value = redrawn(annotation=annotation)
    differs = (value > 32768) != annotation.mask
    return float(np.abs(value[differs] - 32768).max(initial=0))
