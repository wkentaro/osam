import base64
import io

import numpy as np
import PIL.Image

from osam import _requests


def _b64png(*, image: PIL.Image.Image) -> str:
    with io.BytesIO() as f:
        image.save(f, format="PNG")
        return base64.b64encode(f.getvalue()).decode()


def test_segment_request_converts_images_to_rgb() -> None:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    rgb[1, 2] = [0, 0, 255]
    rgba = np.dstack([rgb, np.full((2, 3), 255, np.uint8)])
    palette = PIL.Image.fromarray(rgb).quantize(colors=3)

    images: list[object] = [
        _b64png(image=PIL.Image.fromarray(rgb)),
        _b64png(image=palette),
        _b64png(image=PIL.Image.fromarray(rgba)),
        rgba,
    ]
    for image in images:
        request = _requests.SegmentRequest.model_validate(
            {"model": "sam2:latest", "image": image}
        )
        np.testing.assert_array_equal(request.image, rgb)

    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    request = _requests.SegmentRequest(model="sam2:latest", image=gray)
    np.testing.assert_array_equal(request.image, np.dstack([gray] * 3))
