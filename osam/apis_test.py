import pathlib

import imgviz
import numpy as np
import pytest

from . import apis
from . import types

here = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "model",
    [
        "efficientsam:10m",
        pytest.param("efficientsam:latest", marks=pytest.mark.extra),
        pytest.param("sam:100m", marks=pytest.mark.extra),
        pytest.param("sam:300m", marks=pytest.mark.extra),
        pytest.param("sam:latest", marks=pytest.mark.extra),
        pytest.param("sam2:tiny", marks=pytest.mark.extra),
        pytest.param("sam2:small", marks=pytest.mark.extra),
        pytest.param("sam2:latest", marks=pytest.mark.extra),
        pytest.param("sam2:large", marks=pytest.mark.extra),
    ],
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
    assert annotation.bounding_box is not None


@pytest.mark.parametrize(
    "model, has_mask",
    [
        ("sam3:latest", True),
        pytest.param("yoloworld:latest", False, marks=pytest.mark.extra),
    ],
)
def test_generate_text_to_bounding_box(model: str, has_mask: bool) -> None:
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
        if has_mask:
            assert annotation.mask is not None
            assert annotation.mask.dtype == bool
            assert annotation.mask.shape == image.shape[:2]
        else:
            assert annotation.mask is None


@pytest.mark.parametrize("model", ["sam2:tiny"])
def test_generate_box_to_mask_sam2(model: str) -> None:
    image = imgviz.io.imread(here / "_data" / "dogs.jpg")
    request: types.GenerateRequest = types.GenerateRequest(
        model=model,
        image=image,
        prompt=types.Prompt(
            points=np.array([[1233, 376], [1649, 691]], dtype=np.float32),
            point_labels=np.array([2, 3], dtype=np.int32),
        ),
    )
    response: types.GenerateResponse = apis.generate(request=request)

    assert response.model == model

    assert len(response.annotations) == 1
    for annotation in response.annotations:
        assert annotation.text is None
        assert annotation.score is None
        assert annotation.mask is not None
        assert annotation.mask.dtype == bool
        assert annotation.mask.shape == image.shape[:2]
        assert annotation.bounding_box is not None


@pytest.mark.parametrize("model", ["sam3:latest"])
def test_generate_box_to_mask_sam3(model: str) -> None:
    image = imgviz.io.imread(here / "_data" / "dogs.jpg")
    request: types.GenerateRequest = types.GenerateRequest(
        model=model,
        image=image,
        prompt=types.Prompt(
            points=np.array([[1233, 376], [1649, 691]], dtype=np.float32),
            point_labels=np.array([2, 3], dtype=np.int32),
        ),
    )
    response: types.GenerateResponse = apis.generate(request=request)

    assert response.model == model

    # SAM3 returns multiple mask candidates (unlike SAM2 which returns 1)
    assert len(response.annotations) == 3
    for annotation in response.annotations:
        assert annotation.text == "visual"
        assert annotation.score is not None
        assert annotation.mask is not None
        assert annotation.mask.dtype == bool
        assert annotation.mask.shape == image.shape[:2]
        assert annotation.bounding_box is not None
