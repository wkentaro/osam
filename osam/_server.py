import asyncio
import concurrent.futures
import contextlib
import functools
import logging
import os
from collections.abc import Callable
from typing import Any

import fastapi
import pydantic
from loguru import logger
from pydantic.json_schema import SkipJsonSchema

from . import __version__
from . import apis
from . import types

# apis.generate keeps one model loaded, so inference runs on one thread, in
# order. Queued requests wait here instead of holding FastAPI's thread pool,
# so other routes stay free.
# ponytail: one worker, a model per process if throughput matters.
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


@app.get("/")
def index() -> fastapi.Response:
    return fastapi.Response("osam is running")


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> types.GenerateResponse:
    try:
        return await _run(call=functools.partial(apis.generate, request=request))
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
    except Exception as e:  # e.g. onnxruntime running out of GPU memory
        logger.exception("Failed to generate")
        raise fastapi.HTTPException(500, str(e))


async def _run(*, call: Callable[[], Any]) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_worker, call)
