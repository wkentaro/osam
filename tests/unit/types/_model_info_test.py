import pytest

from osam.types import ModelInfo
from osam.types import ModelMetadata


def _model_info(*, input_resize: str) -> ModelInfo:
    return ModelInfo.model_validate(
        {
            "name": "fake:latest",
            "id": "0",
            "size": None,
            "modified_at": None,
            "metadata": ModelMetadata(license_name="", license_url="", source_url=""),
            "capabilities": ["segment"],
            "prompt_kinds": ["points"],
            "input_size": 1024,
            "input_resize": input_resize,
        }
    )


@pytest.mark.parametrize(
    "input_resize, height, width, expected",
    [
        ("stretch", 3000, 4000, (1024, 1024)),
        ("fit", 3000, 4000, (768, 1024)),
        ("fit", 1599, 2560, (640, 1024)),
        ("fit", 4000, 3000, (1024, 768)),
        ("fit", 100, 100, (1024, 1024)),
        ("fit", 1, 10000, (1, 1024)),
        ("fit", 10000, 1, (1024, 1)),
    ],
)
def test_input_shape(
    input_resize: str, height: int, width: int, expected: tuple[int, int]
) -> None:
    info = _model_info(input_resize=input_resize)
    assert info.input_shape(height=height, width=width) == expected
