import base64
import hashlib
import io
import struct
from typing import Final

import numpy as np
import PIL.Image
import PIL.PngImagePlugin


def image_ndarray_to_png(*, ndarray: np.ndarray) -> bytes:
    # Level 1, not PIL's default 6, for every caller waits on the encoder: it
    # encodes a 2560x1600 image in 0.2 s, not 0.8 s, for 6% more bytes, and
    # osam run --video writes 200 frames at 960x540 25 s sooner.
    with io.BytesIO() as f:
        PIL.Image.fromarray(ndarray).save(f, format="PNG", compress_level=1)
        return f.getvalue()


def image_ndarray_to_b64data(*, ndarray):
    png: bytes = image_ndarray_to_png(ndarray=ndarray)
    return base64.b64encode(png).decode("utf-8")


def image_b64data_to_ndarray(*, b64data, mode: str | None = None):
    # mode, such as "RGB", converts the image, as a palette image's array
    # holds palette indices. ValueError, so pydantic reports bad input as a
    # validation error.
    data = base64.b64decode(b64data)
    try:
        image = PIL.Image.open(io.BytesIO(data))
        return np.asarray(image if mode is None else image.convert(mode))
    # OSError covers a file PIL can't identify and a truncated one.
    except (OSError, PIL.Image.DecompressionBombError) as e:
        raise ValueError(f"not an image file: {e}") from e


MAX_MASK_PIXELS_WITHOUT_IMAGE: Final = 2 * 89_478_485


def max_mask_pixels_without_image() -> int:
    if PIL.Image.MAX_IMAGE_PIXELS is None:
        return MAX_MASK_PIXELS_WITHOUT_IMAGE
    return min(MAX_MASK_PIXELS_WITHOUT_IMAGE, 2 * PIL.Image.MAX_IMAGE_PIXELS)


def _static_png_size(*, png: bytes) -> tuple[int, int]:
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG file")
    sizes: list[tuple[int, int]] = []
    offset = 8
    try:
        while True:
            length, kind = struct.unpack_from(">I4s", png, offset)
            if kind == b"IDAT":
                break
            if kind in (b"acTL", b"fcTL", b"fdAT"):
                raise ValueError("an animated PNG is not a mask")
            if kind == b"IHDR":
                sizes.append(struct.unpack_from(">II", png, offset + 8))
            offset += 12 + length
    except struct.error as e:
        raise ValueError(f"not a PNG file: {e}") from e
    if len(sizes) != 1:
        raise ValueError(f"a PNG has one IHDR, not {len(sizes)}")
    return sizes[0]


def _png_to_ndarray(
    *, png: bytes, within_canvas: bool, mode: str | None = None
) -> np.ndarray:
    try:
        image = (
            PIL.PngImagePlugin.PngImageFile(io.BytesIO(png))
            if within_canvas
            else PIL.Image.open(io.BytesIO(png), formats=["PNG"])
        )
        if mode is not None and image.mode != mode:
            raise ValueError(f"not a PNG of mode {mode}: {image.mode}")
        if (channels := len(image.getbands())) != 1:
            raise ValueError(f"a mask has one channel, not {channels}")
        return np.asarray(image)
    except (
        OSError,
        OverflowError,
        SyntaxError,
        struct.error,
        PIL.Image.DecompressionBombError,
    ) as e:
        raise ValueError(f"not a PNG file: {e}") from e


def scores_b64data_to_ndarray(*, b64data: str) -> np.ndarray:
    png = base64.b64decode(b64data)
    width, height = _static_png_size(png=png)
    if width * height > (limit := max_mask_pixels_without_image()):
        raise ValueError(f"scores size {(width, height)} exceeds {limit} pixels")
    return _png_to_ndarray(png=png, within_canvas=False, mode="L")


def mask_b64data_to_ndarray(
    *, b64data: str, size: tuple[int, int], within_canvas: bool
) -> np.ndarray:
    png = base64.b64decode(b64data)
    png_size = _static_png_size(png=png)
    if png_size != size:
        raise ValueError(f"mask size {png_size} != bounding box size {size}")
    return _png_to_ndarray(png=png, within_canvas=within_canvas) > 0


def hash_image(*, image: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(image).data)
    digest.update(repr((image.shape, image.dtype.str)).encode())
    return digest.hexdigest()


def image_ref(*, image: np.ndarray) -> str:
    # How requests refer to an image the server has seen, in place of the image.
    return f"sha256:{hash_image(image=image)}"
