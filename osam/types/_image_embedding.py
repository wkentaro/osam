from typing import Union

import numpy as np
import numpy.typing as npt
import pydantic


class ImageEmbedding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: npt.NDArray[np.float32]

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
