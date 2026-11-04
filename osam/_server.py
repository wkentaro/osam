import asyncio
import concurrent.futures
import contextlib
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
import pydantic
import starlette.exceptions
import uvicorn
from loguru import logger

from . import __version__
from . import _registry
from . import _runtime as _runtime_module
from . import _types as types
from ._requests import DEFAULT_KEEP_ALIVE
from ._requests import KeepAlive
from ._requests import SegmentRequest
from ._requests import SegmentResponse

# Every call runs on one thread, in order, as the runtime runs one at a time.
# Queued requests wait here instead of holding FastAPI's thread pool, so other
# routes stay free.
# ponytail: one worker, so a long pull also delays segment.
_worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
# `osam serve --device` and `--max-loaded-models` set these. One model by
# default; an application using two, such as one for clicks and one for text,
# sets 2 so they don't reload in turn.
try:
    _runtime = _runtime_module.Runtime(
        device=typing.cast(
            Optional[types.Device], os.environ.get("OSAM_SERVE_DEVICE") or None
        ),
        max_loaded_models=int(os.environ.get("OSAM_MAX_LOADED_MODELS") or 1),
    )
except ValueError as e:
    raise ValueError(f"OSAM_SERVE_DEVICE or OSAM_MAX_LOADED_MODELS: {e}") from None


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
    return ModelList(models=_runtime.list())


@app.get("/api/ps")
def list_running_models() -> RunningModelList:
    return RunningModelList(models=_runtime.ps())


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
            _runtime.pull(model=model_type.name, progress=progress, cancel=cancel)
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
    model_type: type[types.Model] = _get_model_type(name=name)
    await _run(call=functools.partial(_runtime.remove, model=model_type.name))


@app.post("/api/segment", response_model=SegmentResponse)
async def segment(request: SegmentRequest) -> SegmentResponse:
    _get_model_type(name=request.model)
    try:
        return await _run(call=functools.partial(_runtime.segment, request=request))
    except _runtime_module.UnknownImageError as e:
        raise fastapi.HTTPException(412, str(e))
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))


@app.post("/api/load")
async def load(request: LoadRequest) -> None:
    model_type: type[types.Model] = _get_model_type(name=request.model)
    await _run(
        call=functools.partial(
            _runtime.load, model=model_type.name, keep_alive=request.keep_alive
        )
    )


@app.post("/api/generate", deprecated=True)
async def generate(request: LegacyGenerateRequest) -> LegacyGenerateResponse:
    # osam 0.10's route, kept until osam 1.0; /api/segment replaces it.
    # image_embedding is ignored, as the server keeps embeddings itself.
    _warn_generate_deprecated()
    if request.image is None:
        raise fastapi.HTTPException(
            400, "send the image; the server keeps image embeddings itself"
        )
    model_type: type[types.Model] = _get_model_type(name=request.model)
    try:
        response: types.GenerateResponse = await _run(
            call=functools.partial(
                _runtime.generate,
                model=model_type.name,
                image=request.image,
                prompt=request.prompt,
                keep_alive=DEFAULT_KEEP_ALIVE,
            )
        )
    except ValueError as e:
        raise fastapi.HTTPException(400, str(e))
    return LegacyGenerateResponse(
        model=response.model, annotations=response.annotations
    )


@functools.cache
def _warn_generate_deprecated() -> None:
    logger.warning(
        "/api/generate is deprecated and will be removed in osam 1.0; "
        "use /api/segment"
    )


async def _run(*, call: Callable[[], Any]) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_worker, call)


def _get_model_type(*, name: str) -> type[types.Model]:
    try:
        return _registry.get_model_type_by_name(name=name)
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
