import base64
import io

import PIL.Image


def image_ndarray_to_b64data(ndarray):
    pil = PIL.Image.fromarray(ndarray)
    f = io.BytesIO()
    pil.save(f, format="PNG")
    data = f.getvalue()
    return base64.b64encode(data).decode("utf-8")
