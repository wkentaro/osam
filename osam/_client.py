import builtins
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
from typing import Union

import numpy as np
from loguru import logger

from . import __version__
from . import _json
from . import types


class ResponseError(Exception):
    """An error the server returned, with its HTTP status code."""

    status_code: int

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# Client.list shadows list inside the class, hence builtins.list there.
class Client:
    """Run models on an osam server.

    With no host, it starts a private server for this process, which exits
    with it; a frozen application must call multiprocessing.freeze_support()
    first. host, or OSAM_HOST when host is None, connects to a running
    `osam serve` instead, such as "127.0.0.1:11368".

    device, max_loaded_models: as in `osam serve --device` and
    `--max-loaded-models`, for the private server only.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        *,
        device: Optional[types.Device] = None,
        max_loaded_models: Optional[int] = None,
    ):
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
        # The private server's settings, which _server reads on import. Set even
        # when unset, so a value meant for `osam serve` in the caller's
        # environment doesn't reach it.
        self._server_env: dict[str, str] = {
            "OSAM_SERVE_DEVICE": device or "",
            "OSAM_MAX_LOADED_MODELS": str(max_loaded_models or ""),
        }
        self._lock = threading.Lock()
        # host, port and token of the server, swapped whole on a restart, so a
        # request never sends one server's token to another's port.
        self._endpoint: tuple[str, int, Optional[str]] = ("127.0.0.1", 0, None)
        self._https: bool = False
        self._process: Optional[_Process] = None
        # Closing it tells the private server to exit.
        self._pipe: Optional[
            Union[IO[bytes], multiprocessing.connection.Connection]
        ] = None
        # The last image sent and its encoding, as a client often sends one
        # image with several prompts.
        self._image: tuple[str, str] = ("", "")
        # Whether this client owns its server, set once: _process is briefly
        # None while another request restarts it.
        self._private: bool = host is None
        self._closed: bool = False
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
        """Stop the private server, if this client started one."""
        with self._lock:
            self._closed = True
            self._stop()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def version(self) -> str:
        return self._request(method="GET", path="/api/version")["version"]

    def list(self) -> list[types.ModelInfo]:
        """List every model osam knows, pulled or not."""
        return [
            types.ModelInfo.model_validate(model)
            for model in self._request(method="GET", path="/api/models")["models"]
        ]

    def ps(self) -> builtins.list[types.RunningModel]:
        """List the loaded models."""
        return [
            types.RunningModel.model_validate(model)
            for model in self._request(method="GET", path="/api/ps")["models"]
        ]

    def pull(
        self,
        model: str,
        *,
        progress: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        """Download a model, calling progress with each line the server sends.

        Lines look like {"status": "pulling encoder.onnx", "completed": 524288,
        "total": 24799761}.
        """
        for line in self._stream(path="/api/pull", body={"model": model}):
            if "error" in line:
                raise ResponseError(line["error"], status_code=200)
            if progress is not None:
                progress(line)

    def remove(self, model: str) -> None:
        self._request(method="DELETE", path=f"/api/models/{urllib.parse.quote(model)}")

    def generate(
        self,
        model: str,
        *,
        image: Optional[np.ndarray] = None,
        prompt: Optional[types.Prompt] = None,
        annotations: Optional[builtins.list[types.Annotation]] = None,
        keep_alive: Optional[float | str] = None,
    ) -> types.GenerateResponse:
        """Run model on image.

        Prompts on an image the server saw recently reuse its embedding.
        keep_alive: how long the model stays loaded, in seconds or as "5m";
        -1 keeps it loaded, 0 unloads it now. None leaves the server's
        default of five minutes.
        """
        body: dict[str, Any] = {"model": model}
        if prompt is not None:
            body["prompt"] = prompt.model_dump(mode="json")
        if annotations is not None:
            body["annotations"] = [
                annotation.model_dump(mode="json") for annotation in annotations
            ]
        if keep_alive is not None:
            body["keep_alive"] = keep_alive

        def post(**image_fields: str) -> types.GenerateResponse:
            return types.GenerateResponse.model_validate(
                self._request(
                    method="POST", path="/api/generate", body=body | image_fields
                )
            )

        if image is None:
            return post()
        # Hashing takes a fraction of the time PNG encoding does. Read and
        # written whole, so a concurrent call can't mix up two images.
        key: str = _json.hash_image(image=image)
        cached_key, encoded = self._image
        if cached_key == key:
            # Sent before, so the server likely has its embedding; 412 if not,
            # such as for another model or after a restart.
            try:
                return post(image_hash=key)
            except ResponseError as e:
                if e.status_code != 412:
                    raise
        else:
            encoded = _json.image_ndarray_to_b64data(ndarray=image)
            self._image = (key, encoded)
        return post(image=encoded)

    def _request(
        self, *, method: str, path: str, body: Optional[dict[str, Any]] = None
    ) -> Any:
        return self._restarting(
            call=lambda: self._send(method=method, path=path, body=body)
        )

    def _restarting(self, *, call: Callable[[], Any]) -> Any:
        process = self._process
        try:
            return call()
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
        logger.warning("Restarting the osam server after: {error}", error=error)
        with self._lock:
            if self._closed:
                raise RuntimeError("the osam client is closed") from error
            # Unless another request restarted it already.
            if self._process is process:
                self._stop()
                self._start()
        return call()

    def _send(self, *, method: str, path: str, body: Optional[dict[str, Any]]) -> Any:
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
        # Restarts like _request only until the stream starts.
        connection, response = self._restarting(
            call=lambda: self._open_stream(path=path, body=body)
        )
        try:
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
        self, *, method: str, path: str, body: Optional[dict[str, Any]]
    ) -> http.client.HTTPConnection:
        host, port, token = self._endpoint
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        connection = (
            http.client.HTTPSConnection if self._https else http.client.HTTPConnection
        )(host, port)
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
            process, pipe, port = _start_frozen(token=token, env=self._server_env)
        else:
            process, pipe, port = _start_subprocess(token=token, env=self._server_env)
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


_Process = Union[subprocess.Popen, multiprocessing.process.BaseProcess]


def _start_subprocess(
    *, token: str, env: dict[str, str]
) -> tuple[_Process, IO[bytes], int]:
    # A fresh interpreter, unlike multiprocessing, which re-runs the caller's
    # script in the child unless it is guarded by `if __name__ == "__main__"`.
    process = subprocess.Popen(
        [sys.executable, "-c", "from osam._client import _serve_stdio; _serve_stdio()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        process.stdin.write(json.dumps({"token": token, "env": env}).encode())
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
    *, token: str, env: dict[str, str]
) -> tuple[_Process, multiprocessing.connection.Connection, int]:
    # A frozen application has no Python to run, so the server runs from its
    # executable through multiprocessing.freeze_support().
    # Long enough to import onnxruntime from a slow disk.
    START_TIMEOUT: Final = 120

    context = multiprocessing.get_context("spawn")
    conn, child_conn = context.Pipe()
    process = context.Process(
        target=_serve_connection,
        kwargs={"conn": child_conn, "token": token, "env": env},
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


def _wait(*, process: _Process, timeout: Optional[float]) -> bool:
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
    return ResponseError(message, status_code=status_code)


def _serve_stdio() -> None:
    # The child of _start_subprocess: settings on stdin, how it started on
    # stdout, and stdin closing when the parent exits.
    settings: dict[str, Any] = json.loads(sys.stdin.readline())
    # Keep stray output, native libraries' too, off the start line.
    stdout = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def report(message: dict[str, Any]) -> None:
        stdout.write(json.dumps(message) + "\n")
        stdout.flush()

    _serve(
        token=settings["token"],
        env=settings["env"],
        report=report,
        wait_for_parent=sys.stdin.read,
    )


def _serve_connection(
    *,
    conn: multiprocessing.connection.Connection,
    token: str,
    env: dict[str, str],
) -> None:
    def wait_for_parent() -> None:
        try:
            conn.recv()
        except EOFError:
            pass

    _serve(
        token=token,
        env=env,
        report=conn.send,
        wait_for_parent=wait_for_parent,
    )


def _serve(
    *,
    token: str,
    env: dict[str, str],
    report: Callable[[dict[str, Any]], object],
    wait_for_parent: Callable[[], object],
) -> None:
    os.environ.update(env)  # before _server is imported
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
        from . import _server

        _server.run_private(
            token=token, report_port=report_port, wait_for_parent=wait_for_parent
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
