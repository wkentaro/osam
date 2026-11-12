import typing

import pydantic

from ._prompt import Box
from ._prompt import PointLabel
from ._prompt import Prompt

# The actions of /api/track, as ADR 0005 defines them. Coordinates are pixels.


class PromptAction(Prompt):
    # The first action, and only the first: points or box track one object,
    # texts find every match.
    type: typing.Literal["prompt"]
    frame: int = pydantic.Field(ge=0)


class _Edit(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    base: str = pydantic.Field(
        description="The result digest of the response the user saw when making "
        "this action; base_changed with status_code 409 if the result differs"
    )


class AddAction(_Edit):
    type: typing.Literal["add"]
    frame: int = pydantic.Field(ge=0)
    points: list[tuple[float, float]] | None = None
    point_labels: list[PointLabel] | None = None
    box: Box | None = None

    @pydantic.model_validator(mode="after")
    def _check(self) -> "AddAction":
        _check_points(points=self.points, point_labels=self.point_labels)
        if self.points is None and self.box is None:
            raise ValueError("add takes points, a box, or both")
        return self


class RefineAction(_Edit):
    type: typing.Literal["refine"]
    frame: int = pydantic.Field(ge=0)
    object_id: int
    points: list[tuple[float, float]] = pydantic.Field(min_length=1)
    point_labels: list[PointLabel]

    @pydantic.model_validator(mode="after")
    def _check(self) -> "RefineAction":
        _check_points(points=self.points, point_labels=self.point_labels)
        return self


class RemoveAction(_Edit):
    type: typing.Literal["remove"]
    object_id: int


Action = typing.Annotated[
    PromptAction | AddAction | RefineAction | RemoveAction,
    pydantic.Field(discriminator="type"),
]


def _check_points(
    *, points: list[tuple[float, float]] | None, point_labels: list[PointLabel] | None
) -> None:
    if (points is None) != (point_labels is None) or len(points or []) != len(
        point_labels or []
    ):
        raise ValueError("points and point_labels must have the same length")
