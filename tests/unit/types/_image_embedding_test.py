from typing import Any
from typing import cast

import numpy as np
import numpy.typing as npt
import pydantic
import pytest

from osam.types._image_embedding import ImageEmbedding


def test_validate_embedding_rejects_wrong_ndim() -> None:
    with pytest.raises(pydantic.ValidationError) as excinfo:
        ImageEmbedding(
            original_height=1,
            original_width=1,
            embedding=np.zeros((2, 2), dtype=np.float32),
        )
    message = str(excinfo.value)
    assert "embedding must be of dimension 3, not 2" in message
    assert "%r" not in message


def test_validate_embedding_rejects_wrong_dtype() -> None:
    with pytest.raises(pydantic.ValidationError) as excinfo:
        ImageEmbedding(
            original_height=1,
            original_width=1,
            embedding=np.zeros((1, 1, 1), dtype=np.float64),
        )
    message = str(excinfo.value)
    assert "embedding must be of type float" in message
    assert "%r" not in message


@pytest.fixture
def embedding() -> npt.NDArray[np.float32]:
    return np.zeros((1, 1, 1), dtype=np.float32)


def test_validate_extra_features_rejects_non_array_non_dict(
    embedding: npt.NDArray[np.float32],
) -> None:
    extra_features = cast(Any, ["not an array or dict"])
    with pytest.raises(pydantic.ValidationError) as excinfo:
        ImageEmbedding(
            original_height=1,
            original_width=1,
            embedding=embedding,
            extra_features=extra_features,
        )
    message = str(excinfo.value)
    assert "extra_features must be either a numpy array or a dictionary" in message
    assert "%r" not in message


def test_validate_extra_features_rejects_dict_with_wrong_keys(
    embedding: npt.NDArray[np.float32],
) -> None:
    extra_features = cast(Any, [{"data": "", "shape": []}])
    with pytest.raises(pydantic.ValidationError) as excinfo:
        ImageEmbedding(
            original_height=1,
            original_width=1,
            embedding=embedding,
            extra_features=extra_features,
        )
    message = str(excinfo.value)
    assert "extra_features must have keys" in message
    assert "%r" not in message


def test_validate_extra_features_rejects_wrong_ndim(
    embedding: npt.NDArray[np.float32],
) -> None:
    with pytest.raises(pydantic.ValidationError) as excinfo:
        ImageEmbedding(
            original_height=1,
            original_width=1,
            embedding=embedding,
            extra_features=[np.zeros((2, 2), dtype=np.float32)],
        )
    message = str(excinfo.value)
    assert "extra_features must be of dimension 3, not 2" in message
    assert "%r" not in message


def test_validate_extra_features_rejects_wrong_dtype(
    embedding: npt.NDArray[np.float32],
) -> None:
    with pytest.raises(pydantic.ValidationError) as excinfo:
        ImageEmbedding(
            original_height=1,
            original_width=1,
            embedding=embedding,
            extra_features=[np.zeros((1, 1, 1), dtype=np.float64)],
        )
    message = str(excinfo.value)
    assert "extra_features must be of type float" in message
    assert "%r" not in message


def test_json_roundtrip_encodes_arrays_as_base64() -> None:
    embedding = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    original = ImageEmbedding(
        original_height=2,
        original_width=3,
        embedding=embedding,
        extra_features=[embedding * 2],
    )
    dumped = original.model_dump(mode="json")
    assert set(dumped["embedding"]) == {"shape", "dtype", "data"}
    restored = ImageEmbedding.model_validate_json(original.model_dump_json())
    np.testing.assert_array_equal(restored.embedding, embedding)
    np.testing.assert_array_equal(restored.extra_features[0], embedding * 2)


def test_validate_embedding_accepts_nested_lists() -> None:
    restored = ImageEmbedding(
        original_height=1,
        original_width=1,
        embedding=cast(Any, [[[0.5]]]),
    )
    assert restored.embedding.dtype == np.float32
    assert restored.embedding.tolist() == [[[0.5]]]
