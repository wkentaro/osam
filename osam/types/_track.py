import typing

import pydantic

from ._annotation import Annotation

# What /api/track returns, and Client.track.


class TrackFrame(pydantic.BaseModel):
    frame: int
    annotations: list[Annotation]


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
