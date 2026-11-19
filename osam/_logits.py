import typing

import numpy as np
import numpy.typing as npt
import PIL.Image


def mask_from_logits(
    *,
    logits: npt.NDArray[np.float32],
    box: tuple[int, int, int, int],
    height: int,
    width: int,
    threshold: float,
) -> npt.NDArray[np.bool_]:
    logits = shrink_logits(logits=logits, height=height, width=width)
    x0, y0, x1, y1 = box
    scale_x, scale_y = logits.shape[1] / width, logits.shape[0] / height
    image = PIL.Image.fromarray(logits)
    mask = np.empty((y1 - y0, x1 - x0), dtype=bool)
    STRIP_PIXELS: typing.Final = 1 << 22
    rows = max(1, STRIP_PIXELS // (x1 - x0))
    for top in range(y0, y1, rows):
        bottom = min(top + rows, y1)
        mask[top - y0 : bottom - y0] = (
            np.asarray(
                image.resize(
                    (x1 - x0, bottom - top),
                    resample=PIL.Image.Resampling.BILINEAR,
                    box=(x0 * scale_x, top * scale_y, x1 * scale_x, bottom * scale_y),
                )
            )
            > threshold
        )
    return mask


def shrink_logits(
    *, logits: npt.NDArray[np.float32], height: int, width: int
) -> npt.NDArray[np.float32]:
    if logits.shape[0] <= height and logits.shape[1] <= width:
        return logits
    return np.asarray(
        PIL.Image.fromarray(logits).resize(
            (min(logits.shape[1], width), min(logits.shape[0], height)),
            resample=PIL.Image.Resampling.BILINEAR,
        )
    )
