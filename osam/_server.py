import contextlib
import logging
import os
import threading

import fastapi
import pydantic
from loguru import logger
from pydantic.json_schema import SkipJsonSchema

from . import __version__
from . import apis
from . import types

# apis.generate keeps one model loaded, so requests run one at a time.
# ponytail: global lock, a model per worker if throughput matters.
_lock = threading.Lock()


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


# A sync endpoint runs in FastAPI's thread pool, so inference doesn't block the
# event loop.
@app.post("/api/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> types.GenerateResponse:
    with _lock:
        try:
            return apis.generate(request=request)
        except ValueError as e:
            raise fastapi.HTTPException(400, str(e))
        except Exception as e:  # e.g. onnxruntime running out of GPU memory
            logger.exception("Failed to generate")
            raise fastapi.HTTPException(500, str(e))
