import asyncio
import collections
import concurrent.futures
import contextlib
import functools
import json
import logging
import os
import re
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
import pydantic
import starlette.exceptions
import uvicorn
from pydantic.json_schema import SkipJsonSchema

from . import __version__
from . import _json
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
# When the loaded model unloads, as time.time(); None keeps it loaded.
# Written only on _worker.
_expires_at: Optional[float] = None
_unload_timer: Optional[threading.Timer] = None


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


def _parse_duration(value: object) -> object:
    # Seconds, or a duration such as "5m", "1h30m" or "10s", as in Ollama.
    # pydantic validates anything other than a string as a float.
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        pass
    UNITS: Final = {"h": 3600, "m": 60, "s": 1}
    PART: Final = r"(\d+(?:\.\d+)?)([hms])"
    magnitude: str = value.removeprefix("-")
    if not re.fullmatch(f"(?:{PART})+", magnitude):
        raise ValueError(f"invalid duration {value!r}")
    seconds: float = sum(
        float(number) * UNITS[unit] for number, unit in re.findall(PART, magnitude)
    )
    return -seconds if value.startswith("-") else seconds


# Infinite or huge durations would overflow the unload timer; negative ones
# keep the model loaded.
Duration = typing.Annotated[
    float,
    pydantic.BeforeValidator(_parse_duration),
    pydantic.Field(allow_inf_nan=False, le=threading.TIMEOUT_MAX),
]


# Image embeddings stay inside the server: clients send the image each time.
class GenerateRequest(types.GenerateRequest):
    image_embedding: SkipJsonSchema[None] = None
    keep_alive: Duration = pydantic.Field(
        default=300,
        description="How long the model stays loaded after this request, in "
        'seconds or as a duration such as "5m"; negative keeps it loaded, 0 '
        "unloads it now",
    )


class GenerateResponse(pydantic.BaseModel):
    model: str
    annotations: list[types.Annotation]


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
            for model_type in apis.registered_model_types
        ]
    )


@app.get("/api/ps")
def list_running_models() -> RunningModelList:
    model: Optional[types.Model] = apis.running_model
    if model is None:
        return RunningModelList(models=[])
    return RunningModelList(
        models=[
            types.RunningModel(
                name=model.name, device=model.device, expires_at=_expires_at
            )
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


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> types.GenerateResponse:
    model_name: str = _get_model_type(name=request.model).name
    if request.image is None and request.prompt is None and not request.annotations:
        return await _run(
            call=functools.partial(
                _keep_alive_only, request=request, model_name=model_name
            )
        )
    try:
        return await _run(
            call=functools.partial(
                _generate_and_keep_alive, request=request, model_name=model_name
            )
        )
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))


def _keep_alive_only(
    *, request: GenerateRequest, model_name: str
) -> types.GenerateResponse:
    # A request with only a model sets how long that model stays loaded, so
    # keep_alive 0 unloads it, as in Ollama.
    # ponytail: doesn't load an unloaded model; add preloading if a client needs it.
    if apis.running_model is not None and apis.running_model.name == model_name:
        _keep_alive(seconds=request.keep_alive)
    return types.GenerateResponse(model=request.model, annotations=[])


def _generate_and_keep_alive(
    *, request: GenerateRequest, model_name: str
) -> types.GenerateResponse:
    try:
        return _generate_with_cache(request=request, model_name=model_name)
    finally:
        _keep_alive(seconds=request.keep_alive)


def _generate_with_cache(
    *, request: GenerateRequest, model_name: str
) -> types.GenerateResponse:
    # Recent image embeddings, so several prompts on one image encode it once.
    # A SAM3 embedding takes about 100 MB.
    # ponytail: fixed size, make it configurable if memory or hit rate matters.
    EMBEDDING_CACHE_SIZE: Final = 4

    if request.image is None:  # a prompt without an image, which the model rejects
        return apis.generate(request=request, device=_DEVICE)
    key: tuple[str, str] = (model_name, _json.hash_image(image=request.image))
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


def _keep_alive(*, seconds: float) -> None:
    # On _worker. A timer submits the unload back to _worker, which skips it if a
    # later request replaced the timer.
    global _expires_at, _unload_timer

    if _unload_timer is not None:
        _unload_timer.cancel()
        _unload_timer = None
    if seconds < 0:
        _expires_at = None
        return
    if seconds == 0:
        _unload()
        return
    _expires_at = time.time() + seconds
    timer = threading.Timer(
        seconds, lambda: _worker.submit(_unload_if_current, timer=timer)
    )
    timer.daemon = True
    timer.start()
    _unload_timer = timer


def _unload_if_current(*, timer: threading.Timer) -> None:
    if timer is _unload_timer:
        _unload()


def _unload() -> None:
    global _expires_at, _unload_timer
    apis.unload()
    _embeddings.clear()
    _expires_at = None
    _unload_timer = None


async def _run(*, call: Callable[[], Any]) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_worker, call)


def _get_model_type(*, name: str) -> type[types.Model]:
    try:
        return apis.get_model_type_by_name(name=name)
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
        wait_for_parent()
        server.should_exit = True

    threading.Thread(target=exit_with_parent, daemon=True).start()
    report_port(sock.getsockname()[1])
    server.run(sockets=[sock])
