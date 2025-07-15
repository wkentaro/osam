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
    image: Optional[str] = pydantic.Field(default=None)
    prompt: Optional[Prompt] = pydantic.Field(default=None)
    annotations: Optional[list[Annotation]] = pydantic.Field(default=None)


class GenerateResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image_embedding: Optional[ImageEmbedding] = pydantic.Field(default=None)
    annotations: list[Annotation]
