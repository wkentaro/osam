from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pydantic


class Prompt(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    points: Optional[np.ndarray] = pydantic.Field(default=None)
    point_labels: Optional[np.ndarray] = pydantic.Field(default=None)
    texts: Optional[List[str]] = pydantic.Field(default=None)

    iou_threshold: Optional[float] = pydantic.Field(default=0.5)
    score_threshold: Optional[float] = pydantic.Field(default=0.1)
    max_annotations: Optional[int] = pydantic.Field(default=100)

    @pydantic.field_serializer("points")
    def _serialize_points(self: "Prompt", points: np.ndarray) -> List[List[float]]:
        return points.tolist()

    @pydantic.field_serializer("point_labels")
    def _serialize_point_labels(self: "Prompt", point_labels: np.ndarray) -> List[int]:
        return point_labels.tolist()

    @pydantic.field_validator("points", mode="before")
    @classmethod
    def _validate_points(cls: Type, points: Union[list, np.ndarray]):
        if isinstance(points, list):
            points = np.array(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be 2-dimensional")
        if points.shape[1] != 2:
            raise ValueError("points must have 2 columns")
        return points

    @pydantic.field_validator("point_labels", mode="before")
    @classmethod
    def _validate_point_labels(cls: Type, point_labels: Union[list, np.ndarray]):
        if isinstance(point_labels, list):
            point_labels = np.array(point_labels, dtype=int)
        if point_labels.ndim != 1:
            raise ValueError("point_labels must be 1-dimensional")
        if not set(np.unique(point_labels).tolist()).issubset({0, 1, 2, 3}):
            raise ValueError("point_labels must contain only 0, 1, 2, or 3")
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
