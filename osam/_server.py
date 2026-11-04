import asyncio
import collections
import concurrent.futures
import contextlib
import dataclasses
import functools
import json
import logging
import os
import secrets
import socket
import threading
import time
import typing
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Final
from typing import Optional

import fastapi
import numpy as np
import pydantic
import starlette.exceptions
import uvicorn
from loguru import logger

from . import __version__
from . import _apis
from . import _json
from . import _types as types
from ._requests import DEFAULT_KEEP_ALIVE
from ._requests import KeepAlive
from ._requests import SegmentRequest

# Loading and running models, and pull and remove, which must not change the
# files under a model being loaded, all run on one thread, in order.
# Queued requests wait here instead of holding FastAPI's thread pool, so other
# routes stay free.
# ponytail: one worker, so a long pull also delays segment.
_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
# Used only on _worker.
_embeddings: collections.OrderedDict[
    tuple[str, str], types.ImageEmbedding
] = collections.OrderedDict()
# Every loaded model runs on one device, a server setting.
# `osam serve --device` sets it; unset picks one automatically.
_DEVICE: Final = typing.cast(
    Optional[types.Device], os.environ.get("OSAM_SERVE_DEVICE") or None
)
if _DEVICE not in (None, *typing.get_args(types.Device)):
    raise ValueError(f"unknown OSAM_SERVE_DEVICE {_DEVICE!r}")
# One model by default. An application using two, such as one for clicks and
# one for text, sets 2 so they don't reload in turn.
_MAX_LOADED_MODELS: Final = int(os.environ.get("OSAM_MAX_LOADED_MODELS") or 1)
if _MAX_LOADED_MODELS < 1:
    raise ValueError(f"OSAM_MAX_LOADED_MODELS must be 1 or more: {_MAX_LOADED_MODELS}")


@dataclasses.dataclass
class _Loaded:
    model: types.Model
    # When it unloads, as time.time(); None keeps it loaded.
    expires_at: Optional[float] = None
    timer: Optional[threading.Timer] = None


# By model name, least recently used first. Written only on _worker, under
# _loaded_lock, which /api/ps takes to read it from another thread.
_loaded: collections.OrderedDict[str, _Loaded] = collections.OrderedDict()
_loaded_lock = threading.Lock()


@contextlib.asynccontextmanager
async def _lifespan(app: fastapi.FastAPI):
    # https://github.com/tiangolo/fastapi/issues/1508#issuecomment-638365277
    os.makedirs(os.path.expanduser("~/.cache/osam"), exist_ok=True)
    handler = logging.FileHandler(os.path.expanduser("~/.cache/osam/osam.log"))
    handler.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt))
    logging.getLogger("uvicorn").addHandler(handler)
    yield


app: fastapi.FastAPI = fastapi.FastAPI(
    title="osam", version=__version__, lifespan=_lifespan
)


# Errors are {"error": message}, as in Ollama.
def _error(
    *, message: str, status_code: int, headers: Optional[dict[str, str]] = None
) -> fastapi.responses.JSONResponse:
    return fastapi.responses.JSONResponse(
        {"error": message}, status_code=status_code, headers=headers
    )


def _describe_validation_errors(*, errors: Sequence[Any]) -> str:
    return "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
    )


# Starlette's base class also covers routing errors, such as an unknown path.
app.add_exception_handler(
    starlette.exceptions.HTTPException,
    lambda request, exc: _error(
        message=exc.detail, status_code=exc.status_code, headers=exc.headers
    ),
)
# Anything else is a 500, still in that shape; uvicorn logs the traceback.
app.add_exception_handler(
    Exception,
    lambda request, exc: _error(
        message=str(exc) or type(exc).__name__, status_code=500
    ),
)
app.add_exception_handler(
    fastapi.exceptions.RequestValidationError,
    lambda request, exc: _error(
        message=_describe_validation_errors(errors=exc.errors()), status_code=422
    ),
)


class SegmentResponse(pydantic.BaseModel):
    annotations: list[types.Annotation]
    image_hash: str = pydantic.Field(
        description="Of the request's image, to send as image_hash with further "
        "prompts on it",
    )


class LoadRequest(pydantic.BaseModel):
    model: str
    keep_alive: KeepAlive = DEFAULT_KEEP_ALIVE


class LegacyGenerateRequest(types.GenerateRequest):
    # osam 0.10's /api/generate request, with no ndarray in its JSON schema.
    image_embedding: Optional[Any] = pydantic.Field(
        default=None, description="Ignored; the server keeps image embeddings itself"
    )


class LegacyGenerateResponse(types.GenerateResponse):
    image_embedding: None = None


class ModelList(pydantic.BaseModel):
    models: list[types.ModelInfo]


class RunningModelList(pydantic.BaseModel):
    models: list[types.RunningModel]


class Version(pydantic.BaseModel):
    version: str


class PullRequest(pydantic.BaseModel):
    model: str


@app.get("/")
def index() -> fastapi.Response:
    return fastapi.Response("osam is running")


@app.get("/api/version")
def version() -> Version:
    return Version(version=__version__)


@app.get("/api/models")
def list_models() -> ModelList:
    return ModelList(
        models=[
            types.ModelInfo(
                name=model_type.name,
                id=model_type.get_id(),
                size=model_type.get_size(),
                modified_at=model_type.get_modified_at(),
                metadata=model_type.metadata,
            )
            for model_type in _apis.registered_model_types
        ]
    )


@app.get("/api/ps")
def list_running_models() -> RunningModelList:
    # Most recently used first, as Ollama lists them.
    return RunningModelList(
        models=[
            types.RunningModel(
                name=loaded.model.name,
                device=loaded.model.device,
                expires_at=loaded.expires_at,
            )
            for loaded in reversed(_loaded_snapshot())
        ]
    )


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


@app.post("/api/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest) -> SegmentResponse:
    model_type: type[types.Model] = _get_model_type(name=request.model)
    try:
        return await _run(
            call=functools.partial(
                _segment_and_keep_alive, request=request, model_type=model_type
            )
        )
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))


@app.post("/api/load")
async def load(request: LoadRequest) -> None:
    # Loads the model, pulling it if needed, and sets how long it stays loaded;
    # keep_alive 0 unloads it, as in Ollama.
    model_type: type[types.Model] = _get_model_type(name=request.model)

    def work() -> None:
        if request.keep_alive != 0:
            _load(model_name=model_type.name)
        _keep_alive(model_name=model_type.name, seconds=request.keep_alive)

    await _run(call=work)


@app.post("/api/generate", deprecated=True)
async def generate(request: LegacyGenerateRequest) -> LegacyGenerateResponse:
    # osam 0.10's route, kept until osam 1.0; /api/segment replaces it.
    # image_embedding is ignored, as the server keeps embeddings itself.
    _warn_generate_deprecated()
    if request.image is None:
        raise fastapi.HTTPException(
            400, "send the image; the server keeps image embeddings itself"
        )
    image: np.ndarray = request.image
    model_type: type[types.Model] = _get_model_type(name=request.model)

    def work() -> LegacyGenerateResponse:
        try:
            response = _generate_with_cache(
                key=(model_type.name, _json.hash_image(image=image)),
                image=image,
                prompt=request.prompt,
            )
        finally:
            _keep_alive(model_name=model_type.name, seconds=DEFAULT_KEEP_ALIVE)
        return LegacyGenerateResponse(
            model=response.model, annotations=response.annotations
        )

    try:
        return await _run(call=work)
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))


@functools.cache
def _warn_generate_deprecated() -> None:
    logger.warning(
        "/api/generate is deprecated and will be removed in osam 1.0; "
        "use /api/segment"
    )


def _segment_and_keep_alive(
    *, request: SegmentRequest, model_type: type[types.Model]
) -> SegmentResponse:
    try:
        return _segment_with_cache(request=request, model_type=model_type)
    finally:
        _keep_alive(model_name=model_type.name, seconds=request.keep_alive)


def _segment_with_cache(
    *, request: SegmentRequest, model_type: type[types.Model]
) -> SegmentResponse:
    image_hash: str = (
        typing.cast(str, request.image_hash)
        if request.image is None
        else _json.hash_image(image=request.image)
    )
    key: tuple[str, str] = (model_type.name, image_hash)
    embedding: Optional[types.ImageEmbedding] = _embeddings.get(key)
    if request.image is None:
        if embedding is None:
            # Before loading the model, which the client's retry with the image
            # would otherwise load a second time after keep_alive 0 unloads it.
            raise fastapi.HTTPException(
                412, f"no embedding for image {image_hash}; send the image"
            )
        height, width = embedding.original_height, embedding.original_width
    else:
        height, width = request.image.shape[:2]
    prompt: Optional[types.Prompt] = request.to_prompt(
        model_type=model_type, height=height, width=width
    )
    response: types.GenerateResponse = _generate_with_cache(
        key=key, image=request.image, prompt=prompt
    )
    return SegmentResponse(annotations=response.annotations, image_hash=image_hash)


def _generate_with_cache(
    *,
    key: tuple[str, str],
    image: Optional[np.ndarray],
    prompt: Optional[types.Prompt],
) -> types.GenerateResponse:
    # On _worker. Runs prompt on the embedding cached under key, or on image.
    # Recent image embeddings stay cached, so several prompts on one image
    # encode it once.
    # A SAM3 embedding takes about 100 MB.
    # ponytail: fixed size, make it configurable if memory or hit rate matters.
    EMBEDDING_CACHE_SIZE: Final = 4

    # key: the model's name and the image's hash.
    model: types.Model = _load(model_name=key[0])
    embedding: Optional[types.ImageEmbedding] = _embeddings.get(key)
    # Models use the embedding when given one, and return the one they used.
    response: types.GenerateResponse = model.generate(
        request=types.GenerateRequest(
            model=model.name, image=image, image_embedding=embedding, prompt=prompt
        )
    )
    if response.image_embedding is not None:
        _embeddings[key] = response.image_embedding
        _embeddings.move_to_end(key)
        while len(_embeddings) > EMBEDDING_CACHE_SIZE:
            _embeddings.popitem(last=False)
    return response


def _load(*, model_name: str) -> types.Model:
    # On _worker. When the server is full, the least recently used model unloads
    # first; one worker runs one request, so that model is idle.
    if model_name in _loaded:
        with _loaded_lock:
            _loaded.move_to_end(model_name)
        return _loaded[model_name].model
    model_type: type[types.Model] = _apis.get_model_type_by_name(name=model_name)
    # Before unloading anything, so a failed download leaves the others loaded.
    model_type.pull()
    while len(_loaded) >= _MAX_LOADED_MODELS:
        _unload(model_name=next(iter(_loaded)))
    model: Optional[types.Model] = None
    try:
        model = model_type(device=_DEVICE)
    except Exception:
        if not _loaded:
            raise
        logger.exception("Failed to load {}; unloading the other models", model_name)
    if (
        model is not None
        and _DEVICE is None
        and model.device == "cpu"
        and any(loaded.model.device != "cpu" for loaded in _loaded.values())
    ):
        # An automatic device falls back to CPU when the GPU, which the others
        # use, is out of memory.
        logger.warning("{} fell back to CPU; unloading the other models", model_name)
        model = None
    if model is None:
        # Likely out of memory, so unload the others and try once more, as Ollama
        # does. Outside the except block, whose traceback would keep the failed
        # load's sessions, and their memory, alive.
        while _loaded:
            _unload(model_name=next(iter(_loaded)))
        model = model_type(device=_DEVICE)
    with _loaded_lock:
        _loaded[model_name] = _Loaded(model=model)
    return model


def _loaded_snapshot() -> tuple[_Loaded, ...]:
    with _loaded_lock:
        return tuple(_loaded.values())


def _keep_alive(*, model_name: str, seconds: float) -> None:
    # On _worker. A timer submits the unload back to _worker, which skips it if a
    # later request replaced the timer.
    loaded: Optional[_Loaded] = _loaded.get(model_name)
    if loaded is None:  # it failed to load, or it isn't loaded
        return
    if loaded.timer is not None:
        loaded.timer.cancel()
        loaded.timer = None
    if seconds < 0:
        loaded.expires_at = None
        return
    if seconds == 0:
        _unload(model_name=model_name)
        return
    loaded.expires_at = time.time() + seconds
    timer = threading.Timer(
        seconds,
        lambda: _worker.submit(_unload_if_current, model_name=model_name, timer=timer),
    )
    timer.daemon = True
    timer.start()
    loaded.timer = timer


def _unload_if_current(*, model_name: str, timer: threading.Timer) -> None:
    loaded: Optional[_Loaded] = _loaded.get(model_name)
    if loaded is not None and loaded.timer is timer:
        _unload(model_name=model_name)


def _unload(*, model_name: str) -> None:
    with _loaded_lock:
        loaded: _Loaded = _loaded.pop(model_name)
    if loaded.timer is not None:
        loaded.timer.cancel()
    for key in [key for key in _embeddings if key[0] == model_name]:
        del _embeddings[key]


async def _run(*, call: Callable[[], Any]) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_worker, call)


def _get_model_type(*, name: str) -> type[types.Model]:
    try:
        return _apis.get_model_type_by_name(name=name)
    except ValueError as e:
        raise fastapi.HTTPException(404, str(e))


def run_private(
    *,
    token: str,
    report_port: Callable[[int], object],
    wait_for_parent: Callable[[], object],
) -> None:
    # The server osam.Client starts: it serves only requests that carry token,
    # until wait_for_parent returns.
    AUTHORIZATION: Final = f"Bearer {token}"

    @app.middleware("http")
    async def check_token(request: fastapi.Request, call_next):
        given: str = request.headers.get("authorization", "")
        if not secrets.compare_digest(given.encode(), AUTHORIZATION.encode()):
            return fastapi.responses.JSONResponse(
                {"error": "missing or wrong token"}, status_code=401
            )
        return await call_next(request)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()  # so requests sent before uvicorn starts wait instead of failing
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))

    def exit_with_parent() -> None:
        # Shutting down waits for requests in flight, which no one is left to
        # read, so a stuck inference must not keep the process and its GPU
        # memory alive.
        GRACE_SECONDS: Final = 5

        wait_for_parent()
        server.should_exit = True
        time.sleep(GRACE_SECONDS)
        os._exit(0)

    threading.Thread(target=exit_with_parent, daemon=True).start()
    report_port(sock.getsockname()[1])
    server.run(sockets=[sock])
