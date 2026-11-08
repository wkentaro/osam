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


IMAGE_REF_PREFIX: Final = "sha256:"
# An image the server has seen, by the hash of its RGB pixels, as
# _json.image_ref gives it.
ImageRef = typing.Annotated[
    str, pydantic.StringConstraints(pattern=f"^{IMAGE_REF_PREFIX}[0-9a-f]{{64}}$")
]


def _decode_image(*, image: object) -> object:
    # Every model takes RGB, so grayscale, RGBA and palette images become RGB
    # here, before the image is hashed.
    if isinstance(image, str) and not image.startswith(IMAGE_REF_PREFIX):
        return _json.image_b64data_to_ndarray(b64data=image, mode="RGB")
    if not isinstance(image, np.ndarray):
        return image  # pydantic rejects it
    if image.dtype == np.uint8 and (
        image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 4)
    ):
        return np.asarray(PIL.Image.fromarray(image).convert("RGB"))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "image must be a uint8 array of shape (H, W, 3), (H, W, 4) or "
            f"(H, W): got {image.dtype} {image.shape}"
        )
    return image


class SegmentRequest(types.Prompt):
    # The prompt fields, and the model and image to segment.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    image: Base64Image | ImageRef = pydantic.Field(
        description="A base64-encoded image file, such as PNG or JPEG, or the "
        "reference a response gave for one sent before; 412 asks for the file",
    )
    include_masks: bool = pydantic.Field(
        default=True,
        description="false leaves out each annotation's mask, for a client that "
        "needs only boxes and scores",
    )
    keep_alive: KeepAlive = DEFAULT_KEEP_ALIVE

    @pydantic.field_validator("image", mode="before")
    @classmethod
    def _decode(cls, image: object) -> object:
        return _decode_image(image=image)


def to_json_fields(*, fields: dict[str, Any]) -> dict[str, Any]:
    # The fields given, with arrays, and the numbers in them, as JSON types.
    return {
        name: np.asarray(value).tolist()
        for name, value in fields.items()
        if value is not None
    }


class SegmentResponse(pydantic.BaseModel):
    annotations: list[types.Annotation]
    image: ImageRef = pydantic.Field(
        description="A reference to the request's image, to send as image with "
        "further prompts on it",
    )


class TrackRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model: str
    frames: list[Base64Image | ImageRef] = pydantic.Field(
        min_length=1,
        description="The video's frames, each a base64-encoded image file or the "
        "reference to one sent before; 412 asks for the files",
    )
    actions: list[types.Action] = pydantic.Field(
        min_length=1,
        description="Every action the user has taken, in order, starting with a prompt",
    )
    first_frame: int = pydantic.Field(default=0, ge=0)
    last_frame: int | None = pydantic.Field(
        default=None, ge=0, description="The last frame to return; null for the last"
    )
    include_masks: bool = True
    keep_alive: KeepAlive = DEFAULT_KEEP_ALIVE

    @pydantic.field_validator("frames", mode="before")
    @classmethod
    def _decode_frames(cls, frames: object) -> object:
        if not isinstance(frames, list):
            return frames
        return [_decode_image(image=frame) for frame in frames]

    @pydantic.model_validator(mode="after")
    def _check_actions(self) -> "TrackRequest":
        if self.actions[0].type != "prompt" or any(
            a.type == "prompt" for a in self.actions[1:]
        ):
            raise ValueError("the first action is a prompt, and only the first")
        return self
