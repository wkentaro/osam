import base64
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
    data = base64.b64decode(b64data)
    pil = PIL.Image.open(io.BytesIO(data))
    return np.asarray(pil)
