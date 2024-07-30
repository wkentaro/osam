from typing import Optional

import numpy as np
import pydantic

from .. import _json
from ._bounding_box import BoundingBox


class Annotation(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    bounding_box: Optional[BoundingBox] = pydantic.Field(default=None)
    text: Optional[str] = pydantic.Field(default=None)
    score: Optional[float] = pydantic.Field(default=None)
    mask: Optional[np.ndarray] = pydantic.Field(default=None)

    @pydantic.field_validator("mask")
    def validate_mask(cls, mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        if mask.dtype != bool:
            raise ValueError("mask must be boolean arrays")
        return mask

    @pydantic.field_serializer("mask")
    def serialize_mask(self, mask: Optional[np.ndarray]) -> Optional[str]:
        if mask is None:
            return None
        return _json.image_ndarray_to_b64data(ndarray=mask.view(np.uint8) * 255)
