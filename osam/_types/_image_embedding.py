import numpy as np
import numpy.typing as npt
import pydantic


class ImageEmbedding(pydantic.BaseModel):
    # An encoded image, kept in memory by Runtime and never sent over HTTP.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: npt.NDArray[np.float32]
    extra_features: list[npt.NDArray[np.float32]] = pydantic.Field(default_factory=list)
