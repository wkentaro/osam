from typing import Annotated

import numpy as np
import pydantic


class Prompt(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    points: Annotated[
        np.ndarray,
        pydantic.WithJsonSchema(
            {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            }
        ),
    ] | None = pydantic.Field(default=None)
    # point_labels: background=0, foreground=1, box_lt=2, box_rb=3,
    # negative_box_lt=4, negative_box_rb=5
    point_labels: Annotated[
        np.ndarray,
        pydantic.WithJsonSchema(
            {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2, 3, 4, 5]}}
        ),
    ] | None = pydantic.Field(default=None)
    texts: list[str] | None = pydantic.Field(default=None)

    iou_threshold: float | None = pydantic.Field(default=0.5)
    score_threshold: float | None = pydantic.Field(default=0.1)
    max_annotations: int | None = pydantic.Field(default=100, ge=0)

    @pydantic.field_serializer("points")
    def _serialize_points(
        self: "Prompt", points: np.ndarray | None
    ) -> list[list[float]] | None:
        if points is None:
            return None
        return points.tolist()

    @pydantic.field_serializer("point_labels")
    def _serialize_point_labels(
        self: "Prompt", point_labels: np.ndarray | None
    ) -> list[int] | None:
        if point_labels is None:
            return None
        return point_labels.tolist()

    @pydantic.field_validator("points", mode="before")
    @classmethod
    def _validate_points(
        cls: type, points: list | np.ndarray | None
    ) -> np.ndarray | None:
        if points is None:
            return None
        if isinstance(points, list):
            points = np.array(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be 2-dimensional")
        if points.shape[1] != 2:
            raise ValueError("points must have 2 columns")
        return points

    @pydantic.field_validator("point_labels", mode="before")
    @classmethod
    def _validate_point_labels(
        cls: type, point_labels: list | np.ndarray | None
    ) -> np.ndarray | None:
        if point_labels is None:
            return None
        if isinstance(point_labels, list):
            point_labels = np.array(point_labels, dtype=int)
        if point_labels.ndim != 1:
            raise ValueError("point_labels must be 1-dimensional")
        if not set(np.unique(point_labels).tolist()).issubset({0, 1, 2, 3, 4, 5}):
            raise ValueError("point_labels must contain only 0, 1, 2, 3, 4, or 5")
        return point_labels

    @pydantic.model_validator(mode="after")
    def _validate_prompt(self) -> "Prompt":
        if self.points is None and self.point_labels is None:
            if self.texts is None:
                raise ValueError(
                    "texts must be provided when points and point_labels "
                    "are not provided"
                )
        elif (
            self.points is None
            or self.point_labels is None
            or self.points.shape[0] != self.point_labels.shape[0]
        ):
            raise ValueError(
                "points and point_labels must have the same number of rows"
            )
        return self
