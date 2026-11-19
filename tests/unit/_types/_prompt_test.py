import pydantic
import pytest

from osam import _types as types
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


def test_scale_prompt_scales_points_by_center_and_example_boxes_by_edge() -> None:
    prompt = types.Prompt(
        points=[(9.5, 5.5)], point_labels=[1], box=(1.5, 3.5, 9.5, 5.5)
    )

    scaled = types.scale_prompt(prompt=prompt, scale=(0.25, 0.5))

    assert scaled.points == [(2.0, 2.5)]
    assert scaled.box == (0.0, 1.5, 2.0, 2.5)
    boxes = types.Prompt(texts=["dog"], example_boxes=[(4, 2, 8, 6)])
    assert types.scale_prompt(prompt=boxes, scale=(0.25, 0.5)).example_boxes == [
        (1.0, 1.0, 2.0, 3.0)
    ]


def test_scale_prompt_keeps_empty_prompt_lists() -> None:
    points = types.Prompt(points=[], point_labels=[], box=(0, 0, 4, 8))
    assert types.scale_prompt(prompt=points, scale=(0.5, 0.5)).points == []
    boxes = types.Prompt(texts=["dog"], example_boxes=[])
    assert types.scale_prompt(prompt=boxes, scale=(0.5, 0.5)).example_boxes == []
