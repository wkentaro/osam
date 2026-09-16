import pathlib
import re

import imgviz
import numpy as np
import pytest

from osam import apis
from osam import types

image_path = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "_images" / "dogs.jpg"
)


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
    image = imgviz.io.imread(image_path)
    request: types.GenerateRequest = types.GenerateRequest(model=model, image=image)
    response: types.GenerateResponse = apis.generate(request=request)

    assert response.model == model

    assert len(response.annotations) == 1
    annotation: types.Annotation = response.annotations[0]
    assert annotation.text is None
    assert annotation.score is None
    assert annotation.mask is not None
    assert annotation.mask.dtype == bool
    assert annotation.bounding_box is not None
    bb = annotation.bounding_box
    assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)


@pytest.mark.parametrize(
    "model, has_mask",
    [
        ("sam3:latest", True),
        pytest.param("yoloworld:latest", False, marks=pytest.mark.extra),
    ],
)
def test_generate_text_to_bounding_box(model: str, has_mask: bool) -> None:
    image = imgviz.io.imread(image_path)
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
            bb = annotation.bounding_box
            assert annotation.mask.shape == (
                bb.ymax - bb.ymin + 1,
                bb.xmax - bb.xmin + 1,
            )
        else:
            assert annotation.mask is None


@pytest.mark.parametrize("model", ["sam2:tiny"])
def test_generate_box_to_mask_sam2(model: str) -> None:
    image = imgviz.io.imread(image_path)
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
        assert annotation.bounding_box is not None
        bb = annotation.bounding_box
        assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)


@pytest.mark.parametrize("model", ["sam3:latest"])
def test_generate_box_to_mask_sam3(model: str) -> None:
    image = imgviz.io.imread(image_path)
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
        assert annotation.bounding_box is not None
        bb = annotation.bounding_box
        assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)


def test_registered_models_have_immutable_license_metadata() -> None:
    revision_pattern = re.compile(r"/(?:blob|tree)/[0-9a-f]{40}(?:/|$)")

    for model_type in apis.registered_model_types:
        metadata = apis.get_model_metadata(model_type.name)

        assert metadata.license_name
        assert metadata.license_url.startswith("https://")
        assert metadata.source_url.startswith("https://")
        assert revision_pattern.search(metadata.license_url)
        assert revision_pattern.search(metadata.source_url)


def test_yoloworld_metadata_points_to_artifact_provenance() -> None:
    metadata = apis.get_model_metadata("yoloworld")

    assert metadata.license_spdx == "GPL-3.0-only"
    url_prefix = "https://github.com/wkentaro/yolo-world-onnx/blob/"
    assert metadata.license_url.startswith(url_prefix)
    assert metadata.license_url.endswith("/LICENSE")
    assert metadata.source_url.startswith(url_prefix)
    assert metadata.source_url.endswith("/ARTIFACTS.md")

    license_revision = metadata.license_url.removeprefix(url_prefix).split("/", 1)[0]
    source_revision = metadata.source_url.removeprefix(url_prefix).split("/", 1)[0]
    assert license_revision == source_revision
    assert len(source_revision) == 40
    assert all(character in "0123456789abcdef" for character in source_revision)


def test_non_maximum_suppression_runs_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(apis, "_non_maximum_suppression_inference_session", None)

    def fail_pull(*args: object, **kwargs: object) -> None:
        raise AssertionError("The graph ships with the package")

    monkeypatch.setattr(types.Blob, "pull", fail_pull)

    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [52, 52, 62, 62]],
        dtype=np.float32,
    )
    scores = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.6]], dtype=np.float32
    )

    kept_boxes, kept_scores, labels, indices = apis.non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        iou_threshold=0.5,
        score_threshold=0.25,
        max_num_detections=10,
    )

    # Pinned against the graph osam downloaded before it was packaged.
    assert indices.tolist() == [0, 2, 2, 3]
    assert labels.tolist() == [0, 0, 1, 1]
    assert kept_scores.tolist() == pytest.approx([0.9, 0.3, 0.7, 0.6])
    assert kept_boxes.tolist() == boxes[[0, 2, 2, 3]].tolist()
