import asyncio
import concurrent.futures
import contextlib
import functools
import logging
import os
from collections.abc import Callable
from typing import Any
from typing import Optional

import fastapi
import pydantic
from loguru import logger
from pydantic.json_schema import SkipJsonSchema

from . import __version__
from . import apis
from . import types

# apis.generate keeps one model loaded, and pull and remove must not change the
# files under a model being loaded, so they all run on one thread, in order. Queued requests wait here instead of holding FastAPI's thread pool,
# so other routes stay free.
# ponytail: one worker, so a long pull also delays generate.
_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)


@contextlib.asynccontextmanager
async def _lifespan(app: fastapi.FastAPI):
    # https://github.com/tiangolo/fastapi/issues/1508#issuecomment-638365277
    handler = logging.FileHandler(os.path.expanduser("~/.cache/osam/osam.log"))
    handler.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt))
    logging.getLogger("uvicorn").addHandler(handler)
    yield


app: fastapi.FastAPI = fastapi.FastAPI(
    title="osam", version=__version__, lifespan=_lifespan
)


# Image embeddings stay inside the server: clients send the image each time.
class GenerateRequest(types.GenerateRequest):
    image_embedding: SkipJsonSchema[None] = None


class GenerateResponse(pydantic.BaseModel):
    model: str
    annotations: list[types.Annotation]


class ModelInfo(pydantic.BaseModel):
    name: str
    id: str
    size: Optional[int] = pydantic.Field(
        description="Bytes on disk; null if not pulled"
    )
    modified_at: Optional[float] = pydantic.Field(
        description="Unix time of the last download; null if not pulled"
    )
    metadata: types.ModelMetadata


class PullRequest(pydantic.BaseModel):
    model: str


@app.get("/")
def index() -> fastapi.Response:
    return fastapi.Response("osam is running")


@app.get("/api/models")
def list_models() -> list[ModelInfo]:
    return [
        ModelInfo(
            name=model_type.name,
            id=model_type.get_id(),
            size=model_type.get_size(),
            modified_at=model_type.get_modified_at(),
            metadata=model_type.metadata,
        )
        for model_type in apis.registered_model_types
    ]


@app.post("/api/pull")
async def pull(request: PullRequest) -> None:
    await _run(call=_get_model_type(name=request.model).pull)


@app.delete("/api/models/{name}")
async def remove(name: str) -> None:
    await _run(call=_get_model_type(name=name).remove)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> types.GenerateResponse:
    _get_model_type(name=request.model)
    try:
        return await _run(call=functools.partial(apis.generate, request=request))
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
    except Exception as e:  # e.g. onnxruntime running out of GPU memory
        logger.exception("Failed to generate")
        raise fastapi.HTTPException(500, str(e))


async def _run(*, call: Callable[[], Any]) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_worker, call)


def _get_model_type(*, name: str) -> type[types.Model]:
    try:
        return apis.get_model_type_by_name(name=name)
    except ValueError as e:
        raise fastapi.HTTPException(404, str(e))
