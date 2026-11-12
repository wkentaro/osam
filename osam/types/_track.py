import typing

import pydantic

from .._errors import ErrorBody
from ._annotation import Annotation

# What /api/track returns, and Client.track.


class TrackFrame(pydantic.BaseModel):
    frame: int
    annotations: list[Annotation]


class TrackProgress(pydantic.BaseModel):
    progress: typing.Literal[True] = True
    action: int | None = None
    frame: int | None = None
    total_frames: int


class TrackError(ErrorBody):
    status_code: int = pydantic.Field(
        description="400 invalid_request, 404 model_not_found, 409 base_changed "
        "with result, 412 missing_images with images, 500 internal_error, or "
        "502 pull_failed. The HTTP response is already 200."
    )


class TrackObject(pydantic.BaseModel):
    object_id: int
    first_frame: int
    last_frame: int


class TrackDone(pydantic.BaseModel):
    done: typing.Literal[True] = True
    objects: list[TrackObject]
    result: str = pydantic.Field(
        description="A digest of the object ids and masks on every frame, which "
        "the next action gives as base"
    )
    cache: typing.Literal["hit", "partial", "miss"] = pydantic.Field(
        description="hit ran no model, partial applied only the new actions, "
        "and miss replayed every action"
    )
    images: list[str] = pydantic.Field(
        description="A reference to each of the request's frames, to send as "
        "frames with further actions on them"
    )


class TrackResponse(pydantic.BaseModel):
    frames: list[TrackFrame]
    done: TrackDone
