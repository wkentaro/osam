from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pydantic

from .. import _json
from ._annotation import Annotation
from ._image_embedding import ImageEmbedding
from ._prompt import Prompt


class GenerateRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image_embedding: Optional[ImageEmbedding] = pydantic.Field(default=None)
    image: Optional[np.ndarray] = pydantic.Field(default=None)
    prompt: Optional[Prompt] = pydantic.Field(default=None)
    annotations: Optional[List[Annotation]] = pydantic.Field(default=None)

    @pydantic.field_validator("image", mode="before")
    @classmethod
    def validate_image(
        cls, image: Optional[Union[str, np.ndarray]]
    ) -> Optional[np.ndarray]:
        if isinstance(image, str):
            return _json.image_b64data_to_ndarray(b64data=image)
        return image


class GenerateResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image_embedding: Optional[ImageEmbedding] = pydantic.Field(default=None)
    annotations: List[Annotation]
