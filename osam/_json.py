import base64
import hashlib
import io

import numpy as np
import PIL.Image


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


def hash_image(*, image: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(image).data)
    digest.update(repr((image.shape, image.dtype.str)).encode())
    return digest.hexdigest()


def image_ref(*, image: np.ndarray) -> str:
    # How requests refer to an image the server has seen, in place of the image.
    return f"sha256:{hash_image(image=image)}"
