import dataclasses
from typing import List

import numpy as np
import pydantic
from typing_extensions import Annotated


@dataclasses.dataclass
class ImageEmbedding:
    original_height: int
    original_width: int
    embedding: np.ndarray


class Prompt(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    points: Annotated[
        np.ndarray,
        pydantic.BeforeValidator(lambda x: np.asarray(x, dtype=np.float32)),
        pydantic.PlainSerializer(lambda x: x.tolist()),
    ]
    point_labels: Annotated[
        np.ndarray,
        pydantic.BeforeValidator(lambda x: np.asarray(x, dtype=np.int32)),
        pydantic.PlainSerializer(lambda x: x.tolist()),
    ]

    @pydantic.validator("points")
    def validate_points(cls, points):
        if points.ndim != 2:
            raise ValueError("points must be 2-dimensional")
        if points.shape[1] != 2:
            raise ValueError("points must have 2 columns")
        return points

    @pydantic.field_serializer("points")
    def serialize_points(cls, points: np.ndarray) -> List[List[float]]:
        return points.tolist()

    @pydantic.validator("point_labels")
    def validate_point_labels(cls, point_labels, values):
        if point_labels.ndim != 1:
            raise ValueError("point_labels must be 1-dimensional")
        if "points" in values and point_labels.shape[0] != values["points"].shape[0]:
            raise ValueError("point_labels must have the same number of rows as points")
        if not set(np.unique(point_labels).tolist()).issubset({0, 1}):
            raise ValueError("point_labels must contain only 0s and 1s")
        return point_labels

    @pydantic.field_serializer("point_labels")
    def serialize_point_labels(cls, point_labels: np.ndarray) -> List[int]:
        return point_labels.tolist()
