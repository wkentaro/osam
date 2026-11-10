import re

from osam import _registry


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
