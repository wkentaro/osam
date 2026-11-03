import base64
import hashlib
import io

import numpy as np
import PIL.Image


def image_ndarray_to_png(*, ndarray: np.ndarray) -> bytes:
    with io.BytesIO() as f:
        PIL.Image.fromarray(ndarray).save(f, format="PNG")
        return f.getvalue()


def image_ndarray_to_b64data(*, ndarray):
    return base64.b64encode(image_ndarray_to_png(ndarray=ndarray)).decode("utf-8")


def image_b64data_to_ndarray(*, b64data):
    # ValueError, so pydantic reports bad input as a validation error.
    data = base64.b64decode(b64data)
    try:
        return np.asarray(PIL.Image.open(io.BytesIO(data)))
    # OSError covers a file PIL can't identify and a truncated one.
    except (OSError, PIL.Image.DecompressionBombError) as e:
        raise ValueError(f"not an image file: {e}") from e


def hash_image(*, image: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(image).data)
    digest.update(repr((image.shape, image.dtype.str)).encode())
    return digest.hexdigest()
