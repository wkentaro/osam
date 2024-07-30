from typing import List

import numpy as np
import pydantic


class ImageEmbedding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: np.ndarray

    @pydantic.field_validator("embedding")
    def validate_embedding(cls, embedding):
        if embedding.ndim != 3:
            raise ValueError(
                "embedding must be 3-dimensional: (embedding_dim, height, width)"
            )
        return embedding

    @pydantic.field_serializer("embedding")
    def serialize_embedding(self, embedding: np.ndarray) -> List[List[List[float]]]:
        return embedding.tolist()
