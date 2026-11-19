from collections.abc import Mapping
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
    object_id: int | None = pydantic.Field(
        default=None,
        description="The tracked object, numbered in the order objects appear; "
        "null outside /api/track",
    )
    # Serialized as a base64 PNG cropped to bounding_box.
    mask: Base64Image | None = pydantic.Field(default=None)

    @pydantic.field_validator("mask", mode="before")
    @classmethod
    def decode_mask(cls, mask: object, info: pydantic.ValidationInfo) -> object:
        if not isinstance(mask, str):
            return mask
        bb = info.data.get("bounding_box")
        if bb is None:
            raise ValueError("bounding_box is required when mask is provided")
        canvas = (
            info.context.get("canvas_size")
            if isinstance(info.context, Mapping)
            else None
        )
        if canvas is not None and not (
            0 <= bb.xmin
            and 0 <= bb.ymin
            and bb.xmax < canvas[0]
            and bb.ymax < canvas[1]
        ):
            raise ValueError(
                f"bounding box {bb} is outside the {canvas[0]}x{canvas[1]} image"
            )
        size = (bb.xmax - bb.xmin + 1, bb.ymax - bb.ymin + 1)
        limit = _json.max_mask_pixels_without_image()
        if canvas is None and size[0] * size[1] > limit:
            raise ValueError(
                f"mask size {size} exceeds {limit} pixels "
                "without the image's canvas_size in the validation context"
            )
        return _json.mask_b64data_to_ndarray(b64data=mask, size=size)

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
        return _json.image_ndarray_to_b64data(ndarray=mask)
