import pydantic
import pytest

from osam._types._prompt import Prompt


def test_kinds_name_the_prompt_fields_given() -> None:
    assert Prompt().kinds == frozenset()
    assert Prompt(points=[], point_labels=[]).kinds == frozenset()
    assert Prompt(texts=[], example_boxes=[]).kinds == frozenset()
    assert Prompt(points=[(1, 2)], point_labels=[1], box=(0, 0, 1, 1)).kinds == {
        "points",
        "box",
    }
    assert Prompt(texts=["dog"], example_boxes=[(0, 0, 1, 1)]).kinds == {
        "texts",
        "example_boxes",
    }


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"points": [(1, 2)]}, "same length"),
        ({"points": [(1, 2)], "point_labels": [1, 1]}, "same length"),
        ({"example_box_labels": [1]}, "must match example_boxes"),
        ({"points": [(1, 2)], "point_labels": [1], "texts": ["dog"]}, "one or the"),
        ({"box": (0, 0, 1, 1), "example_boxes": [(0, 0, 1, 1)]}, "one or the"),
        ({"points": [(1, 2)], "point_labels": [2]}, "point_labels"),
        ({"texts": ["dog"], "max_annotations": -1}, "max_annotations"),
    ],
)
def test_invalid_prompts_are_rejected(fields: dict, message: str) -> None:
    with pytest.raises(pydantic.ValidationError, match=message):
        Prompt.model_validate(fields)
