import pathlib

import imgviz
import numpy as np
import pytest

import osam

image_path = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "_images" / "dogs.jpg"
)


def test_osam_0_10_calls_still_work_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning, match="osam.types.GenerateRequest"):
        GenerateRequest = osam.types.GenerateRequest
    with pytest.warns(DeprecationWarning, match="osam.types.Prompt"):
        Prompt = osam.types.Prompt
    request = GenerateRequest(
        model="efficientsam:10m",
        image=imgviz.io.imread(image_path),
        prompt=Prompt(points=np.array([[1280, 800]]), point_labels=np.array([1])),
    )
    with pytest.warns(DeprecationWarning, match="use osam.segment"):
        response = osam.apis.generate(request)
    assert response.annotations[0].mask is not None
    assert response.annotations[0].mask.any()
    assert response.image_embedding is not None

    with pytest.warns(DeprecationWarning, match="osam.non_maximum_suppression"):
        boxes, scores, labels, indices = osam.apis.non_maximum_suppression(
            np.array([[0, 0, 10, 10], [1, 1, 10, 10]], dtype=np.float32),
            np.array([[0.9], [0.8]], dtype=np.float32),
            0.5,
            0.1,
            10,
        )
    assert indices.tolist() == [0]

    with pytest.warns(DeprecationWarning):
        assert osam.apis.get_model_metadata("efficientsam:10m").license_name

    with pytest.raises(AttributeError):
        osam.types.Missing  # noqa: B018
