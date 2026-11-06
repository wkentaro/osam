import numpy as np
import pydantic

from .. import _json
from ..types import Annotation
from ..types._annotation import Base64Image
from ._image_embedding import ImageEmbedding
from ._prompt import Prompt


class GenerateRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image_embedding: ImageEmbedding | None = pydantic.Field(default=None)
    # Given in JSON as a base64-encoded image file, such as PNG or JPEG.
    image: Base64Image | None = pydantic.Field(default=None)
    prompt: Prompt | None = pydantic.Field(default=None)

    @pydantic.field_validator("image", mode="before")
    @classmethod
    def validate_image(cls, image: str | np.ndarray | None) -> np.ndarray | None:
        if isinstance(image, str):
            return _json.image_b64data_to_ndarray(b64data=image)
        return image


class GenerateResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image_embedding: ImageEmbedding | None = pydantic.Field(default=None)
    annotations: list[Annotation]
