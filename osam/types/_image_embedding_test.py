from typing import Any
from typing import cast

import numpy as np
import numpy.typing as npt
import pydantic
import pytest

from ._image_embedding import ImageEmbedding


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
