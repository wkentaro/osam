import base64
from typing import Union
from typing import cast

import numpy as np
import numpy.typing as npt
import pydantic


class ImageEmbedding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: npt.NDArray[np.float32]
    extra_features: list[npt.NDArray[np.float32]] = pydantic.Field(default_factory=list)

    @pydantic.field_validator("embedding", mode="before")
    def validate_embedding(
        cls, embedding: Union[npt.NDArray[np.float32], list[list[list[float]]]]
    ) -> npt.NDArray[np.float32]:
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        if embedding.ndim != 3:
            raise ValueError("embedding must be of dimension 3, not %r", embedding.ndim)
        if embedding.dtype != np.float32:
            raise ValueError(
                "embedding must be of type float, but got %r", embedding.dtype
            )
        return embedding

    @pydantic.field_serializer("embedding")
    def serialize_embedding(self, embedding: np.ndarray) -> list[list[list[float]]]:
        return embedding.tolist()

    @pydantic.field_validator("extra_features", mode="before")
    def validate_extra_features(
        cls, extra_features: list[Union[npt.NDArray[np.float32], dict]]
    ) -> list[npt.NDArray[np.float32]]:
        extra_features_ndarray: list[npt.NDArray[np.float32]] = []
        for i, high_res_feature in enumerate(extra_features):
            high_res_feature_ndarray: npt.NDArray[np.float32]
            if isinstance(high_res_feature, np.ndarray):
                high_res_feature_ndarray = cast(
                    npt.NDArray[np.float32], high_res_feature
                )
            elif isinstance(high_res_feature, dict):
                if {"data", "shape", "dtype"} != set(high_res_feature.keys()):
                    raise ValueError(
                        "extra_features must have keys {'data', 'shape', 'dtype'}, "
                        "but got %r",
                        set(high_res_feature.keys()),
                    )
                high_res_feature_ndarray = np.frombuffer(
                    base64.b64decode(high_res_feature["data"]),
                    dtype=high_res_feature["dtype"],
                ).reshape(high_res_feature["shape"])
            else:
                raise ValueError(
                    "extra_features must be either a numpy array or a dictionary, "
                    "but got %r",
                    high_res_feature,
                )
            del high_res_feature

            if high_res_feature_ndarray.ndim != 3:
                raise ValueError(
                    "extra_features must be of dimension 3, not %r",
                    high_res_feature_ndarray.ndim,
                )
            if high_res_feature_ndarray.dtype != np.float32:
                raise ValueError(
                    "extra_features must be of type float, but got %r",
                    high_res_feature_ndarray.dtype,
                )
            extra_features_ndarray.append(high_res_feature_ndarray)
        return extra_features_ndarray

    @pydantic.field_serializer("extra_features")
    def serialize_extra_features(self, extra_features: list[np.ndarray]) -> list[dict]:
        return [_ndarray_to_json(arr=arr) for arr in extra_features]


def _ndarray_to_json(arr: np.ndarray) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "data": base64.b64encode(arr.tobytes()).decode(),
    }
