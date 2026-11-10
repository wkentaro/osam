import pathlib
import re

import imgviz
import pytest

from osam import _registry
from osam import _types as types

image_path = (
    pathlib.Path(__file__).resolve().parents[2] / "examples" / "_images" / "dogs.jpg"
)


# One model at a time, freed before the next loads, as several stay too large
# for a test machine.
_loaded: list[types.Model] = []


def _model(*, name: str) -> types.Model:
    model_type = _registry.get_model_type_by_name(name=name)
    if not _loaded or _loaded[0].name != model_type.name:
        _loaded.clear()
        _loaded.append(model_type())
    return _loaded[0]


def _segment(*, model: str, prompt: types.Prompt) -> list[types.Annotation]:
    loaded = _model(name=model)
    embedding = loaded.encode_image(image=imgviz.io.imread(image_path))
    return loaded.decode(embedding=embedding, prompt=prompt)


@pytest.fixture(scope="module")
def sam3_embedding() -> types.ImageEmbedding:
    # All SAM3 prompts use the same image, so encode it once.
    return _model(name="sam3:latest").encode_image(image=imgviz.io.imread(image_path))


def _sam3(
    *, embedding: types.ImageEmbedding, prompt: types.Prompt
) -> list[types.Annotation]:
    return _model(name="sam3:latest").decode(embedding=embedding, prompt=prompt)


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
    annotations = _segment(model=model, prompt=types.Prompt())

    assert len(annotations) == 1
    annotation: types.Annotation = annotations[0]
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
def test_generate_text_to_bounding_box(
    model: str, has_mask: bool, request: pytest.FixtureRequest
) -> None:
    prompt = types.Prompt(texts=["dog"])
    if model == "sam3:latest":
        annotations = _sam3(
            embedding=request.getfixturevalue("sam3_embedding"), prompt=prompt
        )
    else:
        annotations = _segment(model=model, prompt=prompt)

    assert len(annotations) == 3
    for annotation in annotations:
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
    annotations = _segment(
        model=model,
        prompt=types.Prompt(
            box=(1233, 376, 1649, 691),
        ),
    )

    assert len(annotations) == 1
    for annotation in annotations:
        assert annotation.text is None
        assert annotation.score is None
        assert annotation.mask is not None
        assert annotation.mask.dtype == bool
        assert annotation.bounding_box is not None
        bb = annotation.bounding_box
        assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)


@pytest.mark.extra
@pytest.mark.parametrize("model", ["sam3:latest"])
@pytest.mark.parametrize(
    "example_boxes",
    [
        [(1233, 376, 1649, 691)],
        # two example dogs
        [(1706, 613, 2227, 1438), (450, 551, 1059, 1421)],
    ],
)
def test_generate_example_box_to_mask_sam3(
    model: str,
    example_boxes: list[tuple[float, float, float, float]],
    sam3_embedding: types.ImageEmbedding,
) -> None:
    annotations = _sam3(
        embedding=sam3_embedding,
        prompt=types.Prompt(example_boxes=example_boxes),
    )

    # SAM3 returns multiple mask candidates (unlike SAM2 which returns 1)
    assert len(annotations) == 3
    for annotation in annotations:
        assert annotation.text == "visual"
        assert annotation.score is not None
        assert annotation.mask is not None
        assert annotation.mask.dtype == bool
        assert annotation.bounding_box is not None
        bb = annotation.bounding_box
        assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)


def test_registered_models_have_immutable_license_metadata() -> None:
    revision_pattern = re.compile(r"/(?:blob|tree)/[0-9a-f]{40}(?:/|$)")

    for model_type in _registry.registered_model_types:
        metadata = model_type.metadata

        assert metadata.license_name
        assert metadata.license_url.startswith("https://")
        assert metadata.source_url.startswith("https://")
        assert revision_pattern.search(metadata.license_url)
        assert revision_pattern.search(metadata.source_url)


def test_yoloworld_metadata_points_to_artifact_provenance() -> None:
    metadata = _registry.get_model_type_by_name(name="yoloworld").metadata

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


@pytest.mark.extra
def test_generate_negative_box_sam3(
    sam3_embedding: types.ImageEmbedding,
) -> None:
    # A positive box on one dog and a negative box on another
    annotations = _sam3(
        embedding=sam3_embedding,
        prompt=types.Prompt(
            example_boxes=[(1706, 613, 2227, 1438), (450, 551, 1059, 1421)],
            example_box_labels=[1, 0],
        ),
    )

    [annotation] = annotations
    assert annotation.bounding_box is not None
    assert annotation.bounding_box.xmin > 1059


def test_generate_box_to_mask_sam3(
    sam3_embedding: types.ImageEmbedding,
) -> None:
    # Without text, a box is one object, as in SAM2.
    annotations = _sam3(
        embedding=sam3_embedding,
        prompt=types.Prompt(
            box=(1233, 376, 1649, 691),
        ),
    )

    [annotation] = annotations
    assert annotation.text is None
    assert annotation.score is not None and annotation.score > 0.9
    bb = annotation.bounding_box
    assert bb is not None
    assert 1213 < bb.xmin and bb.xmax < 1669 and 356 < bb.ymin and bb.ymax < 711


def test_generate_max_annotations_sam3(
    sam3_embedding: types.ImageEmbedding,
) -> None:
    annotations = _sam3(
        embedding=sam3_embedding,
        prompt=types.Prompt(texts=["dog"], max_annotations=1),
    )

    [annotation] = annotations
    assert annotation.score is not None and annotation.score > 0.95


def test_generate_negative_box_rejected_by_sam2() -> None:
    with pytest.raises(ValueError, match="takes box, points, not example_boxes"):
        _segment(
            model="sam2:tiny",
            prompt=types.Prompt(
                example_boxes=[(450, 551, 1059, 1421)], example_box_labels=[0]
            ),
        )


def test_generate_box_rejected_by_yoloworld() -> None:
    with pytest.raises(ValueError, match="takes texts, not box"):
        _segment(model="yoloworld", prompt=types.Prompt(box=(450, 551, 1059, 1421)))


def test_generate_point_to_mask_sam3(
    sam3_embedding: types.ImageEmbedding,
) -> None:
    annotations = _sam3(
        embedding=sam3_embedding,
        prompt=types.Prompt(
            points=[(1439, 504), (1439, 1289)],
            point_labels=[1, 1],
        ),
    )

    [annotation] = annotations
    assert annotation.text is None
    assert annotation.score is not None and annotation.score > 0.5
    assert annotation.mask is not None
    assert annotation.bounding_box is not None
    bb = annotation.bounding_box
    assert bb.xmin < 1439 < bb.xmax
    assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)


@pytest.mark.parametrize(
    "limits", [{"score_threshold": 0.99999}, {"max_annotations": 0}]
)
def test_generate_point_filtered_by_sam3(
    limits: dict, sam3_embedding: types.ImageEmbedding
) -> None:
    annotations = _sam3(
        embedding=sam3_embedding,
        prompt=types.Prompt(
            points=[(1439, 504)],
            point_labels=[1],
            **limits,
        ),
    )

    assert annotations == []
