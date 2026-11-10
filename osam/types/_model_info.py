import typing

import pydantic

from ._model_metadata import ModelMetadata

PromptKind = typing.Literal["points", "box", "texts", "example_boxes"]


class ModelInfo(pydantic.BaseModel):
    name: str
    id: str
    size: int | None = pydantic.Field(description="Bytes on disk; null if not pulled")
    modified_at: float | None = pydantic.Field(
        description="Unix time of the last download; null if not pulled"
    )
    metadata: ModelMetadata
    capabilities: list[typing.Literal["segment"]] = pydantic.Field(
        description="The endpoints the model serves, such as segment for /api/segment"
    )
    prompt_kinds: list[PromptKind] = pydantic.Field(
        description="The prompt fields it takes; a request with others gets 400"
    )


class RunningModel(pydantic.BaseModel):
    name: str
    device: str = pydantic.Field(
        description='Device it runs on, such as "cuda", or "cpu" after a fallback'
    )
    expires_at: float | None = pydantic.Field(
        description="Unix time the model unloads; null if it stays loaded"
    )
