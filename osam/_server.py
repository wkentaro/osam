import asyncio
import collections
import concurrent.futures
import contextlib
import functools
import hashlib
import json
import logging
import os
import threading
import typing
from collections.abc import Callable
from typing import Any
from typing import Final
from typing import Optional

import fastapi
import numpy as np
import pydantic
from loguru import logger
from pydantic.json_schema import SkipJsonSchema

from . import __version__
from . import apis
from . import types

# apis.generate keeps one model loaded, and pull and remove must not change the
# files under a model being loaded, so they all run on one thread, in order.
# Queued requests wait here instead of holding FastAPI's thread pool, so other
# routes stay free.
# ponytail: one worker, so a long pull also delays generate.
_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
# Used only on _worker.
_embeddings: collections.OrderedDict[
    tuple[str, str], types.ImageEmbedding
] = collections.OrderedDict()
# One loaded model serves every request, so the device is a server setting.
# `osam serve --device` sets it; unset picks one automatically.
_DEVICE: Final = typing.cast(
    Optional[types.Device], os.environ.get("OSAM_SERVE_DEVICE") or None
)
if _DEVICE not in (None, *typing.get_args(types.Device)):
    raise ValueError(f"unknown OSAM_SERVE_DEVICE {_DEVICE!r}")


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


# Streams NDJSON progress lines, as Ollama does, ending in success or error.
@app.post("/api/pull", response_class=fastapi.responses.StreamingResponse)
async def pull(request: PullRequest) -> fastapi.responses.StreamingResponse:
    model_type: type[types.Model] = _get_model_type(name=request.model)
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    cancel = threading.Event()

    def progress(filename: str, completed: int, total: Optional[int]) -> None:
        line = {"status": f"pulling {filename}", "completed": completed, "total": total}
        loop.call_soon_threadsafe(lines.put_nowait, line)

    def work() -> None:
        try:
            model_type.pull(progress=progress, cancel=cancel)
            last: dict[str, Any] = {"status": "success"}
        except Exception as e:
            last = {"error": str(e)}
        # Through the same queue, so it follows every progress line.
        loop.call_soon_threadsafe(lines.put_nowait, last)

    async def stream():
        loop.run_in_executor(_worker, work)
        try:
            while True:
                line = await lines.get()
                yield json.dumps(line) + "\n"
                if "error" in line or line["status"] == "success":
                    return
        finally:
            cancel.set()  # the client left, so stop downloading

    return fastapi.responses.StreamingResponse(
        stream(), media_type="application/x-ndjson"
    )


@app.delete("/api/models/{name}")
async def remove(name: str) -> None:
    await _run(call=_get_model_type(name=name).remove)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> types.GenerateResponse:
    model_name: str = _get_model_type(name=request.model).name
    try:
        return await _run(
            call=functools.partial(
                _generate_with_cache, request=request, model_name=model_name
            )
        )
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
    except Exception as e:  # e.g. onnxruntime running out of GPU memory
        logger.exception("Failed to generate")
        raise fastapi.HTTPException(500, str(e))


def _generate_with_cache(
    *, request: GenerateRequest, model_name: str
) -> types.GenerateResponse:
    # Recent image embeddings, so several prompts on one image encode it once.
    # A SAM3 embedding takes about 100 MB.
    # ponytail: fixed size, make it configurable if memory or hit rate matters.
    EMBEDDING_CACHE_SIZE: Final = 4

    if request.image is None:
        return apis.generate(request=request, device=_DEVICE)
    key: tuple[str, str] = (model_name, _hash_image(image=request.image))
    if key in _embeddings:
        _embeddings.move_to_end(key)
        return apis.generate(
            request=types.GenerateRequest(
                model=request.model,
                image_embedding=_embeddings[key],
                prompt=request.prompt,
                annotations=request.annotations,
            ),
            device=_DEVICE,
        )
    response: types.GenerateResponse = apis.generate(request=request, device=_DEVICE)
    if response.image_embedding is not None:
        _embeddings[key] = response.image_embedding
        while len(_embeddings) > EMBEDDING_CACHE_SIZE:
            _embeddings.popitem(last=False)
    return response


async def _run(*, call: Callable[[], Any]) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_worker, call)


def _get_model_type(*, name: str) -> type[types.Model]:
    try:
        return apis.get_model_type_by_name(name=name)
    except ValueError as e:
        raise fastapi.HTTPException(404, str(e))


def _hash_image(*, image: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(image).data)
    digest.update(repr((image.shape, image.dtype.str)).encode())
    return digest.hexdigest()
