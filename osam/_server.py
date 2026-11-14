import asyncio
import concurrent.futures
import functools
import json
import logging
import os
import secrets
import socket
import threading
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Final

import fastapi
import pydantic
import starlette.exceptions
import uvicorn

from . import _errors
from . import _runtime
from . import _types as types
from ._requests import LoadRequest
from ._requests import SegmentRequest
from ._requests import SegmentResponse
from ._requests import TrackRequest
from .types import _blob


class ModelList(pydantic.BaseModel):
    models: list[types.ModelInfo]


class RunningModelList(pydantic.BaseModel):
    models: list[types.RunningModel]


class Version(pydantic.BaseModel):
    version: str


class PullRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    model: str


class _TrackStream(fastapi.responses.StreamingResponse):
    media_type = "application/x-ndjson"
    _cancel: threading.Event

    def __init__(self, content, *, cancel: threading.Event):
        super().__init__(content)
        self._cancel = cancel

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cancel.set()


# Errors are {"error": message, "code": code}: the message as in Ollama, and a
# code that stays the same across releases, for clients to branch on.
def _error(
    *,
    message: str,
    status_code: int,
    code: str,
    headers: dict[str, str] | None = None,
    **fields: Any,
) -> fastapi.responses.JSONResponse:
    return fastapi.responses.JSONResponse(
        {"error": message, "code": code, **fields},
        status_code=status_code,
        headers=headers,
    )


def _report(*, exc: Exception) -> dict[str, Any]:
    # The fields of _error for exc. Three errors carry a field more: the images
    # to send again, in place of their references, the result to show before
    # the edit is made again, and the device whose files to pull.
    reported: _errors.Reported = next(
        (
            _errors.REPORTED[t]
            for t in type(exc).__mro__
            if issubclass(t, Exception) and t in _errors.REPORTED
        ),
        _errors.INTERNAL,
    )
    fields: dict[str, Any] = {
        "message": str(exc),
        "status_code": reported.status_code,
        "code": reported.code,
    }
    if reported is _errors.INTERNAL:
        fields["message"] = str(exc) or type(exc).__name__
    if isinstance(exc, _errors.UnknownImageError):
        fields["images"] = exc.refs
    if isinstance(exc, _errors.BaseChangedError):
        fields["result"] = exc.result
    if isinstance(exc, _errors.NotPulledError):
        fields["device"] = exc.device
    return fields


def _describe_validation_errors(*, errors: Sequence[Any]) -> str:
    return "; ".join(
        f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors
    )


def create_app(*, runtime: _runtime.Runtime) -> fastapi.FastAPI:
    # The routes of osam serve, running models on runtime, which the caller
    # closes.
    app = fastapi.FastAPI(title="osam", version=runtime.version())
    # Every call runs on one thread, in order, as the runtime runs one at a time.
    # Queued requests wait here instead of holding FastAPI's thread pool, so other
    # routes stay free. Its idle thread exits once the app is garbage collected.
    # ponytail: one worker, so a long pull also delays segment.
    worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def run(*, call: Callable[[], Any]) -> Any:
        return await asyncio.get_running_loop().run_in_executor(worker, call)

    # Starlette's base class also covers routing errors, such as an unknown path.
    ROUTING_CODES: Final = {404: "not_found", 405: "method_not_allowed"}
    app.add_exception_handler(
        starlette.exceptions.HTTPException,
        lambda request, exc: _error(
            message=exc.detail,
            status_code=exc.status_code,
            code=ROUTING_CODES.get(exc.status_code, _errors.INTERNAL.code),
            headers=exc.headers,
        ),
    )
    # Anything else is a 500, still in that shape; uvicorn logs the traceback.
    for error_type in [Exception, *_errors.REPORTED]:
        app.add_exception_handler(
            error_type, lambda request, exc: _error(**_report(exc=exc))
        )
    app.add_exception_handler(
        fastapi.exceptions.RequestValidationError,
        lambda request, exc: _error(
            message=_describe_validation_errors(errors=exc.errors()),
            status_code=422,
            code=_errors.REPORTED[ValueError].code,
        ),
    )

    @app.get("/")
    def index() -> fastapi.Response:
        return fastapi.Response("osam is running")

    @app.get("/api/version")
    def version() -> Version:
        return Version(version=runtime.version())

    @app.get("/api/models")
    def list_models() -> ModelList:
        return ModelList(models=runtime.list())

    @app.get("/api/ps")
    def list_running_models() -> RunningModelList:
        return RunningModelList(models=runtime.ps())

    # Streams NDJSON progress lines, as Ollama does, ending in success or error.
    @app.post("/api/pull", response_class=fastapi.responses.StreamingResponse)
    async def pull(request: PullRequest) -> fastapi.responses.StreamingResponse:
        # Looked up before streaming, so an unknown model is a 404.
        model_type: type[types.Model] = runtime.model_type(name=request.model)
        loop = asyncio.get_running_loop()
        lines: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        cancel = threading.Event()

        def progress(line: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(lines.put_nowait, line)

        def work() -> None:
            try:
                runtime.pull(model=model_type.name, progress=progress, cancel=cancel)
                last: dict[str, Any] = {"status": "success"}
            except Exception as e:
                last = {
                    "error": str(e),
                    "code": _errors.REPORTED[_errors.DownloadError].code,
                }
            # Through the same queue, so it follows every progress line.
            loop.call_soon_threadsafe(lines.put_nowait, last)

        async def stream():
            loop.run_in_executor(worker, work)
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
        await run(call=functools.partial(runtime.remove, model=name))

    @app.post("/api/segment", response_model=SegmentResponse)
    async def segment(request: SegmentRequest) -> SegmentResponse:
        return await run(call=functools.partial(runtime.segment, request=request))

    @app.post(
        "/api/track",
        status_code=200,
        response_class=_TrackStream,
        responses={
            200: {
                "model": types.TrackProgress
                | types.TrackFrame
                | types.TrackDone
                | types.TrackError,
                "description": "NDJSON progress and finalized frames, ending with "
                "TrackDone or TrackError. Inference errors carry status_code "
                "inside the HTTP 200 stream.",
            },
            422: {
                "description": "invalid_request: request validation failed",
                "content": {
                    "application/json": {
                        "schema": _errors.ErrorBody.model_json_schema()
                    }
                },
            },
        },
    )
    async def track(request: TrackRequest) -> fastapi.responses.Response:
        cancel = threading.Event()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue[
            types.TrackProgress | types.TrackResponse | types.TrackError
        ] = asyncio.Queue()

        def progress(event: types.TrackProgress) -> None:
            # The stream already sent the preparing event.
            if event.frame is not None and not cancel.is_set():
                loop.call_soon_threadsafe(events.put_nowait, event)

        def work() -> None:
            try:
                event = runtime.track(request=request, progress=progress, cancel=cancel)
            except Exception as exc:
                if cancel.is_set() and isinstance(
                    exc, (concurrent.futures.CancelledError, types.PullCancelledError)
                ):
                    return
                fields = _report(exc=exc)
                if fields["status_code"] == _errors.INTERNAL.status_code:
                    logging.getLogger("uvicorn.error").exception(
                        "Video tracking failed"
                    )
                event = types.TrackError(error=fields.pop("message"), **fields)
            if not cancel.is_set():
                loop.call_soon_threadsafe(events.put_nowait, event)

        async def stream():
            # Sent even when another request holds the worker.
            yield (
                types.TrackProgress(total_frames=len(request.frames)).model_dump_json()
                + "\n"
            )
            loop.run_in_executor(worker, work)
            while True:
                event = await events.get()
                if isinstance(event, types.TrackResponse):
                    for frame in event.frames:
                        yield await asyncio.to_thread(frame.model_dump_json) + "\n"
                    yield event.done.model_dump_json() + "\n"
                    return
                yield event.model_dump_json(exclude_none=True) + "\n"
                if isinstance(event, types.TrackError):
                    return

        return _TrackStream(stream(), cancel=cancel)

    @app.post("/api/load")
    async def load(request: LoadRequest) -> types.RunningModel | None:
        return await run(call=functools.partial(runtime.load, request=request))

    return app


def uvicorn_server(*, app: fastapi.FastAPI, **config: Any) -> uvicorn.Server:
    # A server for app that also logs to ~/.cache/osam/osam.log, next to the
    # CLI's log. After uvicorn.Config, which replaces uvicorn's log handlers.
    server = uvicorn.Server(uvicorn.Config(app, **config))
    os.makedirs(os.path.expanduser("~/.cache/osam"), exist_ok=True)
    handler = logging.FileHandler(os.path.expanduser("~/.cache/osam/osam.log"))
    handler.setLevel(logging.DEBUG)
    fmt = "%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt))
    logging.getLogger("uvicorn").addHandler(handler)
    return server


def run_private(
    *,
    token: str,
    runtime: _runtime.Runtime,
    report_port: Callable[[int], object],
    wait_for_parent: Callable[[], object],
) -> None:
    # The server osam.Client starts: it serves only requests that carry token,
    # until wait_for_parent returns.
    AUTHORIZATION: Final = f"Bearer {token}"

    app = create_app(runtime=runtime)

    @app.middleware("http")
    async def check_token(request: fastapi.Request, call_next):
        given: str = request.headers.get("authorization", "")
        if not secrets.compare_digest(given.encode(), AUTHORIZATION.encode()):
            return _error(
                message="missing or wrong token",
                status_code=401,
                code="unauthorized",
            )
        return await call_next(request)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()  # so requests sent before uvicorn starts wait instead of failing
    server = uvicorn_server(app=app, log_level="warning")

    def exit_with_parent() -> None:
        # No one is left to read a request in flight, so the server exits
        # without waiting for it, but only after a cancelled download deletes
        # its partial file.
        CLEANUP_SECONDS: Final = 5

        wait_for_parent()
        _blob.cancel_downloads(timeout=CLEANUP_SECONDS)
        os._exit(0)

    report_port(sock.getsockname()[1])
    threading.Thread(target=exit_with_parent, daemon=True).start()
    server.run(sockets=[sock])
