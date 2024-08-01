import pathlib

import imgviz
import pytest

from . import apis
from . import types

here = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "model",
    ["efficientsam:10m", "efficientsam:latest", "sam:100m", "sam:300m", "sam:latest"],
)
def test_generate_point_to_mask(model: str) -> None:
    image = imgviz.io.imread(here / "_data" / "dogs.jpg")
    request: types.GenerateRequest = types.GenerateRequest(model=model, image=image)
    response: types.GenerateResponse = apis.generate(request=request)

    assert response.model == model

    assert len(response.annotations) == 1
    annotation: types.Annotation = response.annotations[0]
    assert annotation.text is None
    assert annotation.score is None
    assert annotation.mask is not None
    assert annotation.mask.dtype == bool
    assert annotation.mask.shape == image.shape[:2]


def test_generate_text_to_bounding_box() -> None:
    model: str = "yoloworld:latest"

    image = imgviz.io.imread(here / "_data" / "dogs.jpg")
    request: types.GenerateRequest = types.GenerateRequest(
        model=model, image=image, prompt=types.Prompt(texts=["dog"])
    )
    response: types.GenerateResponse = apis.generate(request=request)

    assert response.model == model

    assert len(response.annotations) == 3
    for annotation in response.annotations:
        assert annotation.bounding_box is not None
        assert annotation.text == "dog"
        assert isinstance(annotation.score, float)
        assert annotation.mask is None
