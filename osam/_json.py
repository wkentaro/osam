import base64
import io

import numpy as np
import PIL.Image


def image_ndarray_to_b64data(ndarray):
    pil = PIL.Image.fromarray(ndarray)
    f = io.BytesIO()
    pil.save(f, format="PNG")
    data = f.getvalue()
    return base64.b64encode(data).decode("utf-8")


def image_b64data_to_ndarray(b64data):
    data = base64.b64decode(b64data)
    pil = PIL.Image.open(io.BytesIO(data))
    return np.asarray(pil)
