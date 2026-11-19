from collections.abc import Mapping
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pydantic

from .. import _json
from .. import _logits
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
        description="A base64 PNG of the mask's probability, 0 to 255, on the "
        "model's grid over the whole image; null unless the request asked for it. "
        "Rounding to 8 bits can flip edge pixels of a mask built from it",
    )

    @pydantic.field_validator("mask", mode="before")
    @classmethod
    def decode_mask(cls, mask: object, info: pydantic.ValidationInfo) -> object:
        if not isinstance(mask, str):
            return mask
        bb = info.data.get("bounding_box")
        if bb is None:
            raise ValueError("bounding_box is required when mask is provided")
        canvas = _canvas_size(info=info)
        _check_inside_canvas(bounding_box=bb, canvas=canvas)
        size = (bb.xmax - bb.xmin + 1, bb.ymax - bb.ymin + 1)
        limit = _json.max_mask_pixels_without_image()
        if canvas is None and size[0] * size[1] > limit:
            raise ValueError(
                f"mask size {size} exceeds {limit} pixels "
                "without the image's canvas_size in the validation context"
            )
        return _json.mask_b64data_to_ndarray(
            b64data=mask, size=size, within_canvas=canvas is not None
        )

    @pydantic.field_validator("mask")
    def validate_mask(cls, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        if mask.dtype != bool:
            raise ValueError("mask must be boolean arrays")
        return mask

    @pydantic.field_validator("mask_scores", mode="before")
    @classmethod
    def decode_mask_scores(
        cls, mask_scores: object, info: pydantic.ValidationInfo
    ) -> object:
        if not isinstance(mask_scores, str):
            return mask_scores
        if (bb := info.data.get("bounding_box")) is not None:
            _check_inside_canvas(bounding_box=bb, canvas=_canvas_size(info=info))
        return _json.scores_b64data_to_ndarray(b64data=mask_scores)

    @pydantic.field_validator("mask_scores")
    @classmethod
    def validate_mask_scores(cls, mask_scores: np.ndarray | None) -> np.ndarray | None:
        if mask_scores is not None and (
            mask_scores.dtype != np.uint8
            or mask_scores.ndim != 2
            or not mask_scores.size
        ):
            raise ValueError("mask_scores must be a non-empty 2-D uint8 array")
        return mask_scores

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

    @pydantic.field_serializer("mask", "mask_scores")
    def serialize_image(self, image: np.ndarray | None) -> str | None:
        if image is None:
            return None
        return _json.image_ndarray_to_b64data(ndarray=image)

    def mask_from_scores(
        self, *, bounding_box: BoundingBox, height: int, width: int
    ) -> npt.NDArray[np.bool_]:
        if self.mask_scores is None:
            raise ValueError("mask_scores is unset")
        if height <= 0 or width <= 0:
            raise ValueError(f"size {width}x{height} is not positive")
        _check_inside_canvas(bounding_box=bounding_box, canvas=(width, height))
        if (
            bounding_box.xmin > bounding_box.xmax
            or bounding_box.ymin > bounding_box.ymax
        ):
            raise ValueError(f"bounding box {bounding_box} is inverted")
        return _logits.mask_from_logits(
            logits=self.mask_scores.astype(np.float32) / 255,
            box=(
                bounding_box.xmin,
                bounding_box.ymin,
                bounding_box.xmax + 1,
                bounding_box.ymax + 1,
            ),
            height=height,
            width=width,
            threshold=0.5,
        )


def _canvas_size(*, info: pydantic.ValidationInfo) -> tuple[int, int] | None:
    if not isinstance(info.context, Mapping):
        return None
    return info.context.get("canvas_size")


def _check_inside_canvas(
    *, bounding_box: BoundingBox, canvas: tuple[int, int] | None
) -> None:
    bb = bounding_box
    if canvas is not None and not (
        0 <= bb.xmin and 0 <= bb.ymin and bb.xmax < canvas[0] and bb.ymax < canvas[1]
    ):
        raise ValueError(
            f"bounding box {bb} is outside the {canvas[0]}x{canvas[1]} image"
        )
