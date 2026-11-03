from typing import Optional

import pydantic

from ._model_metadata import ModelMetadata


class ModelInfo(pydantic.BaseModel):
    name: str
    id: str
    size: Optional[int] = pydantic.Field(
        description="Bytes on disk; null if not pulled"
    )
    modified_at: Optional[float] = pydantic.Field(
        description="Unix time of the last download; null if not pulled"
    )
    metadata: ModelMetadata


class RunningModel(pydantic.BaseModel):
    name: str
    device: str = pydantic.Field(
        description='Device it runs on, such as "cuda", or "cpu" after a fallback'
    )
    expires_at: Optional[float] = pydantic.Field(
        description="Unix time the model unloads; null if it stays loaded"
    )
