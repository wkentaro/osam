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
    mask_scores: Base64Image | None = pydantic.Field(
        default=None,
        description="The mask's probabilities x 255 as a uint8 grayscale image "
        "whose pixels tile bounding_box, at the model's own resolution. Resized "
        "bilinearly onto any image size, a pixel is in the mask at 128 or more",
    )

    @pydantic.field_validator("mask_scores", mode="before")
    @classmethod
    def decode_mask_scores(cls, mask_scores: object) -> object:
        if not isinstance(mask_scores, str):
            return mask_scores
        return _json.image_b64data_to_ndarray(b64data=mask_scores, mode="L")

    @pydantic.field_validator("mask_scores")
    @classmethod
    def validate_mask_scores(cls, mask_scores: np.ndarray | None) -> np.ndarray | None:
        if mask_scores is None:
            return None
        if mask_scores.dtype != np.uint8 or mask_scores.ndim != 2:
            raise ValueError(
                f"mask_scores must be a 2D uint8 array: got {mask_scores.dtype} "
                f"{mask_scores.shape}"
            )
        return mask_scores

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
        if self.mask_scores is not None and self.bounding_box is None:
            raise ValueError("bounding_box is required when mask_scores is provided")
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

    @pydantic.field_serializer("mask_scores")
    def serialize_mask_scores(self, mask_scores: np.ndarray | None) -> str | None:
        if mask_scores is None:
            return None
        return _json.image_ndarray_to_b64data(ndarray=mask_scores)
