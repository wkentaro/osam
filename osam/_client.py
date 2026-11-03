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

    device: as in `osam serve --device`, for the private server only.
    """

    def __init__(
        self, host: Optional[str] = None, *, device: Optional[types.Device] = None
    ):
        host = host or os.environ.get("OSAM_HOST") or None
        if host is not None and device is not None:
            raise ValueError("device sets up a private server; host has its own")
        self._device: Optional[types.Device] = device
        self._lock = threading.Lock()
        self._host: str = "127.0.0.1"
        self._port: int = 0
        self._process: Optional[_Process] = None
        # Closing it tells the private server to exit.
        self._pipe: Optional[
            Union[IO[bytes], multiprocessing.connection.Connection]
        ] = None
        self._token: Optional[str] = None
        # The last image sent and its encoding, as a client often sends one
        # image with several prompts.
        self._image: tuple[str, str] = ("", "")
        if host is None:
            self._start()
            return
        url = urllib.parse.urlsplit(host if "://" in host else f"http://{host}")
        self._host = url.hostname or "127.0.0.1"
        self._port = url.port or 11368
        server_version: str = self.version()
        if server_version.split(".")[:2] != __version__.split(".")[:2]:
            raise RuntimeError(
                f"osam {__version__} can't use the server at {host}, which runs "
                f"osam {server_version}; their major and minor versions differ"
            )

    def close(self) -> None:
        """Stop the private server, if this client started one."""
        with self._lock:
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
        if image is not None:
            body["image"] = self._encode_image(image=image)
        if prompt is not None:
            body["prompt"] = prompt.model_dump(mode="json")
        if annotations is not None:
            body["annotations"] = [
                annotation.model_dump(mode="json") for annotation in annotations
            ]
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        return types.GenerateResponse.model_validate(
            self._request(method="POST", path="/api/generate", body=body)
        )

    def _encode_image(self, *, image: np.ndarray) -> str:
        # Hashing takes a fraction of the time PNG encoding does.
        key: str = _json.hash_image(image=image)
        if self._image[0] != key:
            self._image = (key, _json.image_ndarray_to_b64data(ndarray=image))
        return self._image[1]

    def _request(
        self, *, method: str, path: str, body: Optional[dict[str, Any]] = None
    ) -> Any:
        process = self._process
        try:
            return self._send(method=method, path=path, body=body)
        except (OSError, http.client.HTTPException) as e:
            if process is None:
                raise
            error: Exception = e
        except ResponseError as e:
            if process is None or e.status_code != 500:
                raise
            error = e
        # The private server crashed, or failed in a way that can leave it
        # broken, such as a CUDA load that keeps its memory: start a new one.
        logger.warning("Restarting the osam server after: {error}", error=error)
        with self._lock:
            if self._process is process:
                self._stop()
                self._start()
        return self._send(method=method, path=path, body=body)

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
        connection = self._connect(method="POST", path=path, body=body)
        try:
            response = connection.getresponse()
            if response.status >= 400:
                raise _response_error(status_code=response.status, data=response.read())
            for line in response:
                yield json.loads(line)
        finally:
            # Closing it early tells the server to stop, as for an unfinished pull.
            connection.close()

    def _connect(
        self, *, method: str, path: str, body: Optional[dict[str, Any]]
    ) -> http.client.HTTPConnection:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        connection = http.client.HTTPConnection(self._host, self._port)
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
            self._process, self._pipe, port = _start_frozen(
                token=token, device=self._device
            )
        else:
            self._process, self._pipe, port = _start_subprocess(
                token=token, device=self._device
            )
        self._token = token
        self._host, self._port = "127.0.0.1", port

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
    *, token: str, device: Optional[types.Device]
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
        process.stdin.write(json.dumps({"token": token, "device": device}).encode())
        process.stdin.write(b"\n")
        process.stdin.flush()
        # ponytail: no timeout, the child exits rather than hangs on a failed start.
        line: bytes = process.stdout.readline()
    except BaseException:
        process.kill()
        raise
    if not line:
        process.stdin.close()
        raise RuntimeError(f"osam server exited with code {process.wait()} on start")
    return process, process.stdin, int(line)


def _start_frozen(
    *, token: str, device: Optional[types.Device]
) -> tuple[_Process, multiprocessing.connection.Connection, int]:
    # A frozen application has no Python to run, so the server runs from its
    # executable through multiprocessing.freeze_support().
    # Long enough to import onnxruntime from a slow disk.
    START_TIMEOUT: Final = 120

    context = multiprocessing.get_context("spawn")
    conn, child_conn = context.Pipe()
    process = context.Process(
        target=_serve_connection,
        kwargs={"conn": child_conn, "token": token, "device": device},
        name="osam-server",
        daemon=True,
    )
    process.start()
    child_conn.close()
    try:
        if not conn.poll(START_TIMEOUT):
            raise RuntimeError("osam server did not start in time")
        port: int = conn.recv()
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
    # The child of _start_subprocess: settings on stdin, port on stdout, and
    # stdin closing when the parent exits.
    settings: dict[str, Any] = json.loads(sys.stdin.readline())
    stdout = sys.stdout
    sys.stdout = sys.stderr  # keep stray prints off the port line

    def report_port(port: int) -> None:
        stdout.write(f"{port}\n")
        stdout.flush()

    _serve(
        token=settings["token"],
        device=settings["device"],
        report_port=report_port,
        wait_for_parent=sys.stdin.read,
    )


def _serve_connection(
    *,
    conn: multiprocessing.connection.Connection,
    token: str,
    device: Optional[types.Device],
) -> None:
    def wait_for_parent() -> None:
        try:
            conn.recv()
        except EOFError:
            pass

    _serve(
        token=token,
        device=device,
        report_port=conn.send,
        wait_for_parent=wait_for_parent,
    )


def _serve(
    *,
    token: str,
    device: Optional[types.Device],
    report_port: Callable[[int], object],
    wait_for_parent: Callable[[], object],
) -> None:
    if device is not None:
        os.environ["OSAM_SERVE_DEVICE"] = device  # read when _server is imported
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    from . import _server

    _server.run_private(
        token=token, report_port=report_port, wait_for_parent=wait_for_parent
    )
