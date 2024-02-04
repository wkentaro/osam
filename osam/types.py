from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pydantic

from osam import _json


class ImageEmbedding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: np.ndarray

    @pydantic.validator("embedding")
    def validate_embedding(cls, embedding):
        if embedding.ndim != 3:
            raise ValueError(
                "embedding must be 3-dimensional: (embedding_dim, height, width)"
            )
        return embedding


class Prompt(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    points: np.ndarray
    point_labels: np.ndarray

    @pydantic.validator("points", pre=True)
    def validate_points(cls, points):
        if isinstance(points, list):
            points = np.array(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("points must be 2-dimensional")
        if points.shape[1] != 2:
            raise ValueError("points must have 2 columns")
        return points

    @pydantic.field_serializer("points")
    def serialize_points(self, points: np.ndarray) -> List[List[float]]:
        return points.tolist()

    @pydantic.validator("point_labels", pre=True)
    def validate_point_labels(cls, point_labels, values):
        if isinstance(point_labels, list):
            point_labels = np.array(point_labels, dtype=int)
        if point_labels.ndim != 1:
            raise ValueError("point_labels must be 1-dimensional")
        if "points" in values and point_labels.shape[0] != values["points"].shape[0]:
            raise ValueError("point_labels must have the same number of rows as points")
        if not set(np.unique(point_labels).tolist()).issubset({0, 1}):
            raise ValueError("point_labels must contain only 0s and 1s")
        return point_labels

    @pydantic.field_serializer("point_labels")
    def serialize_point_labels(self, point_labels: np.ndarray) -> List[int]:
        return point_labels.tolist()


class GenerateRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image: np.ndarray
    prompt: Optional[Prompt] = pydantic.Field(default=None)

    @pydantic.validator("image", pre=True)
    def validate_image(cls, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, str):
            return _json.image_b64data_to_ndarray(b64data=image)
        return image


class GenerateResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    mask: np.ndarray

    @pydantic.field_serializer("mask")
    def serialize_mask(self, mask: np.ndarray) -> str:
        return _json.image_ndarray_to_b64data(ndarray=mask)
