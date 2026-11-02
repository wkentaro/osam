import collections
import contextlib
import hashlib
import logging
import os
import threading
from typing import Final
from typing import Optional

import fastapi
import numpy as np
import pydantic

from . import __version__
from . import apis
from . import types

# apis.generate keeps one model loaded, and pull and remove must not change the
# files under a model being loaded, so requests run one at a time.
# ponytail: global lock, so a long pull also delays generate.
_lock = threading.Lock()
_embeddings: collections.OrderedDict[
    tuple[str, str], types.ImageEmbedding
] = collections.OrderedDict()


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


class GenerateRequest(types.GenerateRequest):
    device: Optional[types.Device] = pydantic.Field(
        default=None, description="Execution device; omitted picks one automatically."
    )
    return_image_embedding: bool = pydantic.Field(
        default=False,
        description="Include image_embedding in the response, to send back later "
        "instead of the image.",
    )


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
def pull(request: PullRequest) -> None:
    model_type = _get_model_type(name=request.model)
    with _lock:
        model_type.pull()


@app.delete("/api/models/{name}")
def remove(name: str) -> None:
    model_type = _get_model_type(name=name)
    with _lock:
        model_type.remove()


@app.post("/api/generate")
def generate(request: GenerateRequest) -> types.GenerateResponse:
    # Recent image embeddings, so several prompts on one image encode it once.
    # A SAM3 embedding takes about 100 MB.
    # ponytail: fixed size, make it configurable if memory or hit rate matters.
    EMBEDDING_CACHE_SIZE: Final = 4

    model_name: str = _get_model_type(name=request.model).name
    key: Optional[tuple[str, str]] = None
    if request.image is not None and request.image_embedding is None:
        key = (model_name, _hash_image(image=request.image))
    with _lock:
        if key is not None and key in _embeddings:
            _embeddings.move_to_end(key)
            request = request.model_copy(
                update={"image_embedding": _embeddings[key], "image": None}
            )
        try:
            response: types.GenerateResponse = apis.generate(
                request=request, device=request.device
            )
        except ValueError as e:
            raise fastapi.HTTPException(400, str(e))
        except RuntimeError as e:  # e.g. the device is unavailable
            raise fastapi.HTTPException(500, str(e))
        if key is not None and response.image_embedding is not None:
            _embeddings[key] = response.image_embedding
            while len(_embeddings) > EMBEDDING_CACHE_SIZE:
                _embeddings.popitem(last=False)
    if not request.return_image_embedding:
        response.image_embedding = None
    return response


def _get_model_type(*, name: str) -> type[types.Model]:
    try:
        return apis.get_model_type_by_name(name=name)
    except ValueError as e:
        raise fastapi.HTTPException(404, str(e))


def _hash_image(*, image: np.ndarray) -> str:
    digest = hashlib.sha256(np.ascontiguousarray(image).data)
    digest.update(repr((image.shape, image.dtype.str)).encode())
    return digest.hexdigest()
