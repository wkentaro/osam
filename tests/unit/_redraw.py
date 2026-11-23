import math

import numpy as np

from osam import types


def _lerp(*, grid: np.ndarray, t: np.ndarray, axis: int) -> np.ndarray:
    t = np.clip(t, 0, grid.shape[axis] - 1)
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, grid.shape[axis] - 1)
    w1 = np.expand_dims(t - i0, 1 - axis)
    return np.take(grid, i0, axis) * (1 - w1) + np.take(grid, i1, axis) * w1


def _pixels_centered_in(
    *, edges: tuple[float, float, float, float], image_size: tuple[int, int]
) -> types.BoundingBox:
    xmin, ymin, xmax, ymax = edges
    return types.BoundingBox(
        xmin=max(0, math.ceil(xmin - 0.5)),
        ymin=max(0, math.ceil(ymin - 0.5)),
        xmax=min(image_size[0] - 1, math.floor(xmax - 0.5)),
        ymax=min(image_size[1] - 1, math.floor(ymax - 0.5)),
    )


def redrawn(
    *, annotation: types.Annotation, image_size: tuple[int, int], scale: int = 1
) -> tuple[np.ndarray, types.BoundingBox]:
    assert annotation.mask_logits is not None
    assert annotation.mask_logits_box is not None
    image_size = (image_size[0] * scale, image_size[1] * scale)
    xmin, ymin, xmax, ymax = (scale * edge for edge in annotation.mask_logits_box)
    area = _pixels_centered_in(edges=(xmin, ymin, xmax, ymax), image_size=image_size)
    if annotation.text is not None:
        assert annotation.bounding_box is not None
        bb = annotation.bounding_box
        clip = _pixels_centered_in(
            edges=(
                scale * bb.xmin,
                scale * bb.ymin,
                scale * (bb.xmax + 1),
                scale * (bb.ymax + 1),
            ),
            image_size=image_size,
        )
        area = types.BoundingBox(
            xmin=max(area.xmin, clip.xmin),
            ymin=max(area.ymin, clip.ymin),
            xmax=min(area.xmax, clip.xmax),
            ymax=min(area.ymax, clip.ymax),
        )
    grid = annotation.mask_logits.astype(np.float64)
    rows, cols = grid.shape
    xs = (np.arange(area.xmin, area.xmax + 1) + 0.5 - xmin) * cols / (xmax - xmin) - 0.5
    ys = (np.arange(area.ymin, area.ymax + 1) + 0.5 - ymin) * rows / (ymax - ymin) - 0.5
    return _lerp(grid=_lerp(grid=grid, t=ys, axis=0), t=xs, axis=1), area


def distance_where_redraw_differs(
    *, annotation: types.Annotation, image_size: tuple[int, int]
) -> float:
    assert annotation.bounding_box is not None
    assert annotation.mask is not None
    value, area = redrawn(annotation=annotation, image_size=image_size)
    bb = annotation.bounding_box
    mask = np.zeros(value.shape, bool)
    mask[
        bb.ymin - area.ymin : bb.ymax + 1 - area.ymin,
        bb.xmin - area.xmin : bb.xmax + 1 - area.xmin,
    ] = annotation.mask
    differs = (value > 32768) != mask
    return float(np.abs(value[differs] - 32768).max(initial=0))
