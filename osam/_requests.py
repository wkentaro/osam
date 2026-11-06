import re
import threading
import typing
from collections.abc import Sequence
from typing import Any
from typing import Final

import numpy as np
import PIL.Image
import pydantic

from . import _json
from . import _types as types
from .types._annotation import Base64Image


def _parse_duration(value: object) -> object:
    # Seconds, or a duration such as "5m", "1h30m" or "10s", as in Ollama.
    # pydantic validates anything other than a string as a float.
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        pass
    UNITS: Final = {"h": 3600, "m": 60, "s": 1}
    PART: Final = r"(\d+(?:\.\d+)?)([hms])"
    magnitude: str = value.removeprefix("-")
    if not re.fullmatch(f"(?:{PART})+", magnitude):
        raise ValueError(f"invalid duration {value!r}")
    seconds: float = sum(
        float(number) * UNITS[unit] for number, unit in re.findall(PART, magnitude)
    )
    return -seconds if value.startswith("-") else seconds


# Infinite or huge durations would overflow the unload timer; negative ones
# keep the model loaded.
Duration = typing.Annotated[
    float,
    pydantic.BeforeValidator(_parse_duration),
    pydantic.Field(allow_inf_nan=False, le=threading.TIMEOUT_MAX),
]


# Five minutes, as in Ollama.
DEFAULT_KEEP_ALIVE: Final = 300
KeepAlive = typing.Annotated[
    Duration,
    pydantic.Field(
        description="How long the model stays loaded after this request, in "
        'seconds or as a duration such as "5m"; negative keeps it loaded, 0 '
        "unloads it now",
    ),
]
Box = tuple[float, float, float, float]
PointLabel = typing.Literal[0, 1]


class LoadRequest(pydantic.BaseModel):
    model: str
    keep_alive: KeepAlive = DEFAULT_KEEP_ALIVE


class SegmentRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image: Base64Image | None = pydantic.Field(
        default=None,
        description="A base64-encoded image file, such as PNG or JPEG",
    )
    image_hash: str | None = pydantic.Field(
        default=None,
        description="In place of image, the hash of an image sent before, whose "
        "embedding the server may still have; 412 asks for the image itself",
    )
    # One object: points with at most one box.
    points: list[tuple[float, float]] | None = pydantic.Field(
        default=None, description="x, y in pixels"
    )
    point_labels: list[PointLabel] | None = pydantic.Field(
        default=None, description="1 for foreground, 0 for background"
    )
    box: Box | None = pydantic.Field(
        default=None, description="xmin, ymin, xmax, ymax in pixels"
    )
    # Every match: texts, example boxes, or both.
    texts: list[str] | None = None
    example_boxes: list[Box] | None = pydantic.Field(
        default=None, description="xmin, ymin, xmax, ymax in pixels"
    )
    example_box_labels: list[PointLabel] | None = pydantic.Field(
        default=None,
        description="1 for a positive example, 0 for a negative one; all 1 if unset",
    )
    iou_threshold: float | None = None
    score_threshold: float | None = None
    max_annotations: int | None = pydantic.Field(default=None, ge=0)
    keep_alive: KeepAlive = DEFAULT_KEEP_ALIVE

    @pydantic.field_validator("image", mode="before")
    @classmethod
    def _decode_image(cls, image: object) -> object:
        # Every model takes RGB, so grayscale, RGBA and palette images become
        # RGB here, before the image is hashed.
        if isinstance(image, str):
            return _json.image_b64data_to_ndarray(b64data=image, mode="RGB")
        if (
            isinstance(image, np.ndarray)
            and image.dtype == np.uint8
            and (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 4))
        ):
            return np.asarray(PIL.Image.fromarray(image).convert("RGB"))
        return image

    @pydantic.model_validator(mode="after")
    def _check(self) -> "SegmentRequest":
        if self.image is not None and self.image_hash is not None:
            raise ValueError("send image or image_hash, not both")
        if self.image is None and self.image_hash is None:
            raise ValueError("send image or image_hash")
        if (self.points is None) != (self.point_labels is None) or len(
            self.points or []
        ) != len(self.point_labels or []):
            raise ValueError("points and point_labels must have the same length")
        if self.example_box_labels is not None and len(self.example_box_labels) != len(
            self.example_boxes or []
        ):
            raise ValueError("example_box_labels must match example_boxes")
        if (self.points is not None or self.box is not None) and (
            self.texts is not None or self.example_boxes is not None
        ):
            raise ValueError(
                "points and box segment one object, and texts and example_boxes "
                "find every match; give one or the other"
            )
        return self

    def to_prompt(
        self, *, model_type: type[types.Model], height: int, width: int
    ) -> types.Prompt | None:
        # The one place that builds the models' point labels: 2 and 3 for a
        # box's corners, 4 and 5 for a negative box's.
        thresholds: dict[str, Any] = self.model_dump(
            include={"iou_threshold", "score_threshold", "max_annotations"},
            exclude_none=True,
        )
        points: list[Sequence[float]] = list(self.points or [])
        labels: list[int] = list(self.point_labels or [])
        if self.box is not None:
            points += [self.box[:2], self.box[2:]]
            labels += [2, 3]
        texts: list[str] | None = self.texts
        if self.example_boxes is not None:
            for example, label in zip(
                self.example_boxes,
                self.example_box_labels or [1] * len(self.example_boxes),
            ):
                points += [example[:2], example[2:]]
                labels += [2, 3] if label else [4, 5]
            # SAM3's text for "every object like the boxes".
            texts = texts or ["visual"]
        if not points and texts is None:
            if not thresholds:
                return None  # the model's default prompt
            return model_type.default_prompt(height=height, width=width).model_copy(
                update=thresholds
            )
        return types.Prompt(
            points=np.array(points, dtype=float) if points else None,
            point_labels=np.array(labels, dtype=int) if labels else None,
            texts=texts,
            **thresholds,
        )


def to_json_fields(*, fields: dict[str, Any]) -> dict[str, Any]:
    # The fields given, with arrays, and the numbers in them, as JSON types.
    return {
        name: np.asarray(value).tolist()
        for name, value in fields.items()
        if value is not None
    }


class SegmentResponse(pydantic.BaseModel):
    annotations: list[types.Annotation]
    image_hash: str = pydantic.Field(
        description="Of the request's image, to send as image_hash with further "
        "prompts on it",
    )
