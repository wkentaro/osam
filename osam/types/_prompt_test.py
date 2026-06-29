import json

import numpy as np
import pydantic
import pytest

from ._prompt import Prompt


def test_model_dump_with_texts_only_serializes_none_points() -> None:
    dumped = Prompt(texts=["dog"]).model_dump()
    assert dumped["points"] is None
    assert dumped["point_labels"] is None
    assert dumped["texts"] == ["dog"]


def test_model_dump_json_with_texts_only_serializes_null_points() -> None:
    dumped = json.loads(Prompt(texts=["dog"]).model_dump_json())
    assert dumped["points"] is None
    assert dumped["point_labels"] is None


def test_model_dump_serializes_points_and_labels_as_lists() -> None:
    dumped = Prompt(
        points=np.array([[1.0, 2.0]]), point_labels=np.array([1])
    ).model_dump()
    assert dumped["points"] == [[1.0, 2.0]]
    assert dumped["point_labels"] == [1]


def test_texts_only_roundtrips_through_model_dump() -> None:
    src = Prompt(texts=["dog"])
    restored = Prompt.model_validate(src.model_dump())
    assert restored.texts == ["dog"]
    assert restored.points is None
    assert restored.point_labels is None


def test_texts_only_roundtrips_through_model_dump_json() -> None:
    src = Prompt(texts=["dog"])
    restored = Prompt.model_validate_json(src.model_dump_json())
    assert restored.texts == ["dog"]
    assert restored.points is None
    assert restored.point_labels is None


def test_points_roundtrip_through_model_dump() -> None:
    prompt = Prompt(
        points=np.array([[1.0, 2.0], [3.0, 4.0]]), point_labels=np.array([1, 0])
    )
    restored = Prompt.model_validate(prompt.model_dump())
    np.testing.assert_array_equal(restored.points, prompt.points)
    np.testing.assert_array_equal(restored.point_labels, prompt.point_labels)


def test_points_without_point_labels_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        Prompt(points=np.array([[1.0, 2.0]]))


def test_point_labels_outside_allowed_set_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError):
        Prompt(points=np.array([[1.0, 2.0]]), point_labels=np.array([7]))
