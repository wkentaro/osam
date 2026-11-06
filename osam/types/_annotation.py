from typing import Annotated

import numpy as np
import pydantic

from .. import _json
from ._bounding_box import BoundingBox

# An image array that JSON carries as a base64-encoded image file.
Base64Image = Annotated[
    np.ndarray,
    pydantic.WithJsonSchema({"type": "string", "contentEncoding": "base64"}),
]


class Annotation(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    bounding_box: BoundingBox | None = pydantic.Field(default=None)
    text: str | None = pydantic.Field(default=None)
    score: float | None = pydantic.Field(default=None)
    # Serialized as a base64 PNG cropped to bounding_box.
    mask: Base64Image | None = pydantic.Field(default=None)

    @pydantic.field_validator("mask", mode="before")
    @classmethod
    def decode_mask(cls, mask: object) -> object:
        # Reads the base64 PNG serialize_mask writes.
        if isinstance(mask, str):
            return _json.image_b64data_to_ndarray(b64data=mask) > 0
        return mask

    @pydantic.field_validator("mask")
    def validate_mask(cls, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        if mask.dtype != bool:
            raise ValueError("mask must be boolean arrays")
        return mask

    @pydantic.model_validator(mode="after")
    def validate_mask_bbox_consistency(self) -> "Annotation":
        if self.mask is not None:
            if self.bounding_box is None:
                raise ValueError("bounding_box is required when mask is provided")
            bb = self.bounding_box
            expected = (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)
            if self.mask.shape != expected:
                raise ValueError(
                    f"mask shape {self.mask.shape} != expected bbox shape {expected}"
                )
        return self

    @pydantic.field_serializer("mask")
    def serialize_mask(self, mask: np.ndarray | None) -> str | None:
        if mask is None:
            return None
        return _json.image_ndarray_to_b64data(ndarray=mask.view(np.uint8) * 255)
