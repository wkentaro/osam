import builtins
import contextlib
import http.client
import json
import multiprocessing
import multiprocessing.connection
import os
import secrets
import subprocess
import sys
import threading
import urllib.parse
from collections.abc import Callable
from collections.abc import Iterator
from typing import IO
from typing import Any
from typing import Final
from typing import Optional

import numpy as np
from loguru import logger

from . import __version__
from . import _json
from . import _requests
from . import types


class ResponseError(Exception):
    """An error the server returned, with its HTTP status code."""

    status_code: int

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class _InvalidRequestError(ResponseError, ValueError):
    # A request the server rejected, with 400 or 422, such as a prompt that
    # mixes one object with every match: a ValueError, as in a local call.
    pass


class _NotFoundError(ResponseError, LookupError):
    # A 404, such as an unknown model: a LookupError, as in a local call.
    pass


class _PullError(ResponseError, RuntimeError):
    # A pull that failed on the server: a RuntimeError, as in a local call.
    pass


# HttpBackend.list shadows list inside the class, hence builtins.list there.
class HttpBackend:
    # Client's backend for a server: a private one it starts, which exits with
    # this process, or a running osam serve at host. See Client for the
    # arguments.

    _timeout: float | None
    # The private server's Runtime arguments.
    _server_settings: dict[str, Any]
    _lock: threading.Lock
    # host, port and token of the server, swapped whole on a restart, so a
    # request never sends one server's token to another's port.
    _endpoint: tuple[str, int, str | None]
    _https: bool
    _process: "_Process | None"
    # Closing it tells the private server to exit.
    _pipe: IO[bytes] | multiprocessing.connection.Connection | None
    # The last image sent: its local hash, its encoding, and the hash the
    # server returned for it, as a client often sends one image with
    # several prompts.
    _image: tuple[str, str, str]
    # Whether this client owns its server, set once: _process is briefly
    # None while another request restarts it.
    _private: bool
    _closed: bool

    def __init__(
        self,
        host: str | None = None,
        *,
        device: types.Device | None = None,
        max_loaded_models: int | None = None,
        timeout: float | None = None,
    ):
        self._timeout = timeout
        host = host or os.environ.get("OSAM_HOST") or None
        if host is not None and (device, max_loaded_models) != (None, None):
            raise ValueError(
                "device and max_loaded_models set up a private server; "
                "host has its own"
            )
        if max_loaded_models is not None and max_loaded_models < 1:
            raise ValueError(
                f"max_loaded_models must be 1 or more: {max_loaded_models}"
            )
        self._server_settings = {
            "device": device,
            "max_loaded_models": max_loaded_models or 1,
        }
        self._lock = threading.Lock()
        self._endpoint = ("127.0.0.1", 0, None)
        self._https = False
        self._process = None
        self._pipe = None
        self._image = ("", "", "")
        self._private = host is None
        self._closed = False
        if host is None:
            self._start()
            return
        url = urllib.parse.urlsplit(host if "://" in host else f"http://{host}")
        if url.scheme not in ("http", "https"):
            raise ValueError(f"host must be an http or https URL: {host!r}")
        self._https = url.scheme == "https"
        self._endpoint = (
            url.hostname or "127.0.0.1",
            url.port or (443 if self._https else 11368),
            None,
        )
        server_version: str = self.version()
        if server_version.split(".")[:2] != __version__.split(".")[:2]:
            raise RuntimeError(
                f"osam {__version__} can't use the server at {host}, which runs "
                f"osam {server_version}; their major and minor versions differ"
            )

    def close(self) -> None:
        # Stops the private server, if this backend started one.
        with self._lock:
            self._closed = True
            self._stop()

    def version(self) -> str:
        return self._request(method="GET", path="/api/version")["version"]

    def list(self) -> list[types.ModelInfo]:
        return [
            types.ModelInfo.model_validate(model)
            for model in self._request(method="GET", path="/api/models")["models"]
        ]

    def ps(self) -> builtins.list[types.RunningModel]:
        return [
            types.RunningModel.model_validate(model)
            for model in self._request(method="GET", path="/api/ps")["models"]
        ]

    def pull(
        self,
        *,
        model: str,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        for line in self._stream(path="/api/pull", body={"model": model}):
            if "error" in line:
                raise _PullError(line["error"], status_code=200)
            if progress is not None:
                progress(line)

    def remove(self, *, model: str) -> None:
        self._request(method="DELETE", path=f"/api/models/{urllib.parse.quote(model)}")

    def load(self, *, request: _requests.LoadRequest) -> None:
        self._request(method="POST", path="/api/load", body=request.model_dump())

    def segment(
        self, *, request: _requests.SegmentRequest
    ) -> _requests.SegmentResponse:
        assert request.image is not None
        image: np.ndarray = request.image
        body: dict[str, Any] = request.model_dump(
            mode="json", exclude={"image", "image_hash"}, exclude_none=True
        )

        def post(**image_fields: str) -> _requests.SegmentResponse:
            response = _requests.SegmentResponse.model_validate(
                self._request(
                    method="POST", path="/api/segment", body=body | image_fields
                )
            )
            self._image = (key, encoded, response.image_hash)
            return response

        # Hashing takes a fraction of the time PNG encoding does. Read and
        # written whole, so a concurrent call can't mix up two images.
        key: str = _json.hash_image(image=image)
        cached_key, encoded, server_hash = self._image
        if cached_key != key:
            encoded = _json.image_ndarray_to_b64data(ndarray=image)
        elif server_hash:
            # Sent before, so the server likely has its embedding; 412 if not,
            # such as for another model or after a restart.
            try:
                return post(image_hash=server_hash)
            except ResponseError as e:
                if e.status_code != 412:
                    raise
        return post(image=encoded)

    def _request(
        self, *, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        return self._restarting(
            call=lambda: self._send(method=method, path=path, body=body)
        )

    def _restarting(self, *, call: Callable[[], Any]) -> Any:
        process = self._process
        try:
            with self._restart_on_timeout():
                return call()
        except TimeoutError:
            raise  # an OSError, but not one to retry
        except (OSError, http.client.HTTPException) as e:
            if not self._private:
                raise
            error: Exception = e
        except ResponseError as e:
            if not self._private or e.status_code != 500:
                raise
            error = e
        # The private server crashed, or failed in a way that can leave it
        # broken, such as a CUDA load that keeps its memory: start a new one.
        self._restart(process=process, error=error)
        with self._restart_on_timeout():
            return call()

    @contextlib.contextmanager
    def _restart_on_timeout(self) -> Iterator[None]:
        # Stops a hung private server and re-raises; the request would likely
        # time out again, so it isn't retried.
        process = self._process
        try:
            yield
        except TimeoutError as e:
            if self._private:
                self._restart(process=process, error=e)
            raise

    def _restart(self, *, process: Optional["_Process"], error: Exception) -> None:
        logger.warning("Restarting the osam server after: {error}", error=error)
        with self._lock:
            if self._closed:
                raise RuntimeError("the osam client is closed") from error
            # Unless another request restarted it already.
            if self._process is process:
                self._stop()
                self._start()

    def _send(self, *, method: str, path: str, body: dict[str, Any] | None) -> Any:
        connection = self._connect(method=method, path=path, body=body)
        try:
            response = connection.getresponse()
            data: bytes = response.read()
        finally:
            connection.close()
        if response.status >= 400:
            raise _response_error(status_code=response.status, data=data)
        return json.loads(data) if data else None

    def _stream(self, *, path: str, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        # Retries like _request only until the stream starts; a timeout after
        # that restarts a private server and raises.
        connection, response = self._restarting(
            call=lambda: self._open_stream(path=path, body=body)
        )
        try:
            with self._restart_on_timeout():
                for line in response:
                    yield json.loads(line)
        finally:
            # Closing it early tells the server to stop, as for an unfinished pull.
            connection.close()

    def _open_stream(
        self, *, path: str, body: dict[str, Any]
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        connection = self._connect(method="POST", path=path, body=body)
        try:
            response = connection.getresponse()
            if response.status >= 400:
                raise _response_error(status_code=response.status, data=response.read())
        except BaseException:
            connection.close()
            raise
        return connection, response

    def _connect(
        self, *, method: str, path: str, body: dict[str, Any] | None
    ) -> http.client.HTTPConnection:
        host, port, token = self._endpoint
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        connection = (
            http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        )(host, port, timeout=self._timeout)
        connection.request(
            method,
            path,
            body=None if body is None else json.dumps(body),
            headers=headers,
        )
        return connection

    def _start(self) -> None:
        token: str = secrets.token_urlsafe(32)
        if getattr(sys, "frozen", False):
            process, pipe, port = _start_frozen(
                token=token, settings=self._server_settings
            )
        else:
            process, pipe, port = _start_subprocess(
                token=token, settings=self._server_settings
            )
        # The endpoint first: a request that sees the new process must also see
        # its endpoint, or it would restart a healthy server.
        self._endpoint = ("127.0.0.1", port, token)
        self._process, self._pipe = process, pipe

    def _stop(self) -> None:
        if self._process is None:
            return
        assert self._pipe is not None
        self._pipe.close()
        if not _wait(process=self._process, timeout=10):
            self._process.kill()
            _wait(process=self._process, timeout=None)
        self._process = self._pipe = None


_Process = subprocess.Popen | multiprocessing.process.BaseProcess


def _start_subprocess(
    *, token: str, settings: dict[str, Any]
) -> tuple[_Process, IO[bytes], int]:
    # A fresh interpreter, unlike multiprocessing, which re-runs the caller's
    # script in the child unless it is guarded by `if __name__ == "__main__"`.
    process = subprocess.Popen(
        [sys.executable, "-c", "from osam._http import _serve_stdio; _serve_stdio()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(json.dumps({"token": token, "settings": settings}).encode())
        process.stdin.write(b"\n")
        process.stdin.flush()
        # ponytail: no timeout, the child exits rather than hangs on a failed start.
        line: bytes = process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"osam server exited with code {process.wait()} on start"
            )
        port: int = _started_port(message=json.loads(line))
    except BaseException:
        process.kill()
        process.stdin.close()
        raise
    return process, process.stdin, port


def _start_frozen(
    *, token: str, settings: dict[str, Any]
) -> tuple[_Process, multiprocessing.connection.Connection, int]:
    # A frozen application has no Python to run, so the server runs from its
    # executable through multiprocessing.freeze_support().
    # Long enough to import onnxruntime from a slow disk.
    START_TIMEOUT: Final = 120

    context = multiprocessing.get_context("spawn")
    conn, child_conn = context.Pipe()
    process = context.Process(
        target=_serve_connection,
        kwargs={"conn": child_conn, "token": token, "settings": settings},
        name="osam-server",
        daemon=True,
    )
    process.start()
    child_conn.close()
    try:
        if not conn.poll(START_TIMEOUT):
            raise RuntimeError("osam server did not start in time")
        port: int = _started_port(message=conn.recv())
    except BaseException as e:
        process.kill()
        conn.close()
        if isinstance(e, EOFError):
            raise RuntimeError(
                f"osam server exited with code {process.exitcode} on start"
            ) from None
        raise
    return process, conn, port


def _wait(*, process: _Process, timeout: float | None) -> bool:
    # True once the process exited.
    if isinstance(process, subprocess.Popen):
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True
    process.join(timeout=timeout)
    return not process.is_alive()


def _response_error(*, status_code: int, data: bytes) -> ResponseError:
    try:
        message: str = json.loads(data)["error"]
    except (ValueError, KeyError, TypeError):
        message = data.decode(errors="replace")
    if status_code in (400, 422):
        return _InvalidRequestError(message, status_code=status_code)
    if status_code == 404:
        return _NotFoundError(message, status_code=status_code)
    return ResponseError(message, status_code=status_code)


def _serve_stdio() -> None:
    # The child of _start_subprocess: settings on stdin, how it started on
    # stdout, and stdin closing when the parent exits.
    start: dict[str, Any] = json.loads(sys.stdin.readline())
    # Keep stray output, native libraries' too, off the start line.
    stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def report(message: dict[str, Any]) -> None:
        stdout.write(json.dumps(message) + "\n")
        stdout.flush()

    _serve(
        token=start["token"],
        settings=start["settings"],
        report=report,
        wait_for_parent=sys.stdin.read,
    )


def _serve_connection(
    *,
    conn: multiprocessing.connection.Connection,
    token: str,
    settings: dict[str, Any],
) -> None:
    def wait_for_parent() -> None:
        try:
            conn.recv()
        except EOFError:
            pass

    _serve(
        token=token,
        settings=settings,
        report=conn.send,
        wait_for_parent=wait_for_parent,
    )


def _serve(
    *,
    token: str,
    settings: dict[str, Any],
    report: Callable[[dict[str, Any]], object],
    wait_for_parent: Callable[[], object],
) -> None:
    # A windowed frozen application, such as a PyInstaller --noconsole build,
    # has no standard streams, and logging to them fails.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    started: bool = False

    def report_port(port: int) -> None:
        nonlocal started
        started = True
        report({"port": port})

    try:
        from . import _runtime
        from . import _server

        _server.run_private(
            token=token,
            runtime=_runtime.Runtime(**settings),
            report_port=report_port,
            wait_for_parent=wait_for_parent,
        )
    except Exception as e:  # e.g. a broken onnxruntime install
        if started:
            raise
        # To the parent, which reports it, rather than a traceback on stderr.
        report({"error": f"{type(e).__name__}: {e}"})
        sys.exit(1)


def _started_port(*, message: dict[str, Any]) -> int:
    if "error" in message:
        raise RuntimeError(f"osam server failed to start: {message['error']}")
    return message["port"]
