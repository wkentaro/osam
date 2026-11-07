import re
import threading
import typing
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


class LoadRequest(pydantic.BaseModel):
    model: str
    keep_alive: KeepAlive = DEFAULT_KEEP_ALIVE


class SegmentRequest(types.Prompt):
    # The prompt fields, and the model and image to segment.
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
        return self


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
