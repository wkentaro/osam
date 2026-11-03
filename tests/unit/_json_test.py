import base64

import numpy as np
import pytest

from osam import _json


@pytest.fixture
def image():
    y, x = np.meshgrid(np.arange(10), np.arange(10))
    center = (np.array(x.shape) - 1) / 2
    image_float = np.exp(-((x - center[1]) ** 2 + (y - center[0]) ** 2) / 10)
    image = (image_float * 255).astype(np.uint8)
    return image


def test_image_ndarray_to_b64data(image):
    b64data = _json.image_ndarray_to_b64data(ndarray=image)
    assert isinstance(b64data, str)
    # The PNG signature; its length varies with the pillow and zlib version.
    assert base64.b64decode(b64data).startswith(b"\x89PNG\r\n\x1a\n")


def test_image_b64data_to_ndarray(image):
    b64data = _json.image_ndarray_to_b64data(ndarray=image)
    image_recovered = _json.image_b64data_to_ndarray(b64data=b64data)
    np.testing.assert_array_equal(image, image_recovered)
