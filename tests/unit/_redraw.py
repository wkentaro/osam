import numpy as np

from osam import types


def _lerp(*, grid: np.ndarray, t: np.ndarray, axis: int) -> np.ndarray:
    t = np.clip(t, 0, grid.shape[axis] - 1)
    i0 = np.floor(t).astype(np.int64)
    i1 = np.minimum(i0 + 1, grid.shape[axis] - 1)
    w1 = np.expand_dims(t - i0, 1 - axis)
    return np.take(grid, i0, axis) * (1 - w1) + np.take(grid, i1, axis) * w1


def _samples(
    *,
    start: float,
    stop: float,
    clip: tuple[float, float],
    size: int,
    scale: float,
    cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = round(size * scale)
    centers = (np.arange(count) + 0.5) * size / count
    pixels = np.flatnonzero(
        (max(start, clip[0]) <= centers) & (centers < min(stop, clip[1]))
    )
    return pixels, (centers[pixels] - start) / (stop - start) * cells - 0.5


def redrawn(
    *, annotation: types.Annotation, image_size: tuple[int, int], scale: float = 1
) -> tuple[np.ndarray, types.BoundingBox]:
    assert annotation.mask_logits is not None
    assert annotation.mask_logits_box is not None
    xmin, ymin, xmax, ymax = annotation.mask_logits_box
    clip = (0, 0, image_size[0], image_size[1])
    if annotation.text is not None:
        assert annotation.bounding_box is not None
        bb = annotation.bounding_box
        clip = (bb.xmin, bb.ymin, bb.xmax + 1, bb.ymax + 1)
    grid = annotation.mask_logits.astype(np.float64)
    rows, cols = grid.shape
    xs, tx = _samples(
        start=xmin,
        stop=xmax,
        clip=(clip[0], clip[2]),
        size=image_size[0],
        scale=scale,
        cells=cols,
    )
    ys, ty = _samples(
        start=ymin,
        stop=ymax,
        clip=(clip[1], clip[3]),
        size=image_size[1],
        scale=scale,
        cells=rows,
    )
    area = types.BoundingBox(
        xmin=int(xs[0]), ymin=int(ys[0]), xmax=int(xs[-1]), ymax=int(ys[-1])
    )
    return _lerp(grid=_lerp(grid=grid, t=ty, axis=0), t=tx, axis=1), area


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
