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
    mask_logits: Base64Image | None = pydantic.Field(
        default=None,
        description="The model's own logit cells around bounding_box, as a "
        "16-bit grayscale image of 32768 + 256 x logit clipped to 0..65535. "
        "Resized bilinearly so that the cells tile mask_logits_box, a pixel is "
        "in the mask above 32768",
    )
    mask_logits_box: tuple[float, float, float, float] | None = pydantic.Field(
        default=None,
        description="xmin, ymin, xmax, ymax in pixels of mask_logits' outer "
        "cell edges, which can reach past the image",
    )

    @pydantic.field_validator("mask_logits", mode="before")
    @classmethod
    def decode_mask_logits(cls, mask_logits: object) -> object:
        if not isinstance(mask_logits, str):
            return mask_logits
        return _json.logits_b64data_to_ndarray(b64data=mask_logits)

    @pydantic.field_validator("mask_logits")
    @classmethod
    def validate_mask_logits(cls, mask_logits: np.ndarray | None) -> np.ndarray | None:
        if mask_logits is None:
            return None
        if (
            mask_logits.dtype != np.uint16
            or mask_logits.ndim != 2
            or not mask_logits.size
        ):
            raise ValueError(
                f"mask_logits must be a 2D uint16 array: got {mask_logits.dtype} "
                f"{mask_logits.shape}"
            )
        return mask_logits

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
        return _json.mask_b64data_to_ndarray(
            b64data=mask, size=(bb.xmax - bb.xmin + 1, bb.ymax - bb.ymin + 1)
        )

    @pydantic.field_validator("mask")
    def validate_mask(cls, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        if mask.dtype != bool:
            raise ValueError("mask must be boolean arrays")
        return mask

    @pydantic.model_validator(mode="after")
    def validate_mask_bbox_consistency(self) -> "Annotation":
        if (self.mask_logits is None) != (self.mask_logits_box is None):
            raise ValueError("mask_logits and mask_logits_box come together")
        if self.mask_logits_box is not None:
            xmin, ymin, xmax, ymax = self.mask_logits_box
            if not (
                np.isfinite(self.mask_logits_box).all() and xmin < xmax and ymin < ymax
            ):
                raise ValueError(f"mask_logits_box {self.mask_logits_box} is empty")
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

    @pydantic.field_serializer("mask_logits")
    def serialize_mask_logits(self, mask_logits: np.ndarray | None) -> str | None:
        if mask_logits is None:
            return None
        return _json.image_ndarray_to_b64data(ndarray=mask_logits)
