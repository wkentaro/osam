import contextlib
import multiprocessing
import pathlib
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from typing import Final
from typing import Iterator

import numpy as np
import PIL.Image
import pytest
import uvicorn

import osam
from osam import _client
from osam import _server

_IMAGE: Final = np.asarray(
    PIL.Image.open(
        pathlib.Path(__file__).resolve().parents[2] / "examples/_images/dogs.jpg"
    )
)


@pytest.fixture(scope="module")
def client():
    with osam.Client() as client:
        yield client


def test_private_server_segments(client: osam.Client) -> None:
    assert client.version() == osam.__version__
    assert "efficientsam:10m" in {model.name for model in client.list()}

    (annotation,) = client.segment(
        "efficientsam:10m", _IMAGE, points=np.array([[1280, 800]]), point_labels=[1]
    )
    assert annotation.mask is not None
    assert annotation.mask.any()
    assert [model.name for model in client.ps()] == ["efficientsam:10m"]
    client.load("efficientsam:10m", keep_alive=0)
    assert client.ps() == []

    # Prompts the model doesn't take, as a local call would raise them.
    prompts: list[dict[str, Any]] = [
        {"points": [[1280, 800]], "point_labels": [1], "texts": ["dog"]},
        {"example_boxes": [[0, 0, 9, 9]]},
        {"texts": ["dog"]},
    ]
    for prompt in prompts:
        with pytest.raises(ValueError):
            client.segment("efficientsam:10m", _IMAGE, **prompt)

    with pytest.raises(osam.ResponseError) as e:
        client.segment("missing", _IMAGE)
    assert e.value.status_code == 404
    assert "not found" in str(e.value)


def test_private_server_needs_its_token(client: osam.Client) -> None:
    connection = _client.http.client.HTTPConnection(*client._endpoint[:2])
    connection.request("GET", "/api/version")
    assert connection.getresponse().status == 401


def test_private_server_restarts_after_a_crash(client: osam.Client) -> None:
    assert client._process is not None
    for call in [client.version, lambda: client.pull("efficientsam:10m")]:
        assert client._process is not None
        client._process.kill()
        _client._wait(process=client._process, timeout=None)
        call()


def test_concurrent_requests_survive_one_restart(client: osam.Client) -> None:
    assert client._process is not None
    client._process.kill()
    _client._wait(process=client._process, timeout=None)
    versions: list[str] = []
    threads = [
        threading.Thread(target=lambda: versions.append(client.version()))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert versions == [osam.__version__] * 8


def test_private_server_exits_with_its_parent(tmp_path: pathlib.Path) -> None:
    # No `if __name__ == "__main__"`, as in a quick script.
    script = (
        "import os, osam\n"
        "client = osam.Client()\n"
        "print(client._endpoint[1], flush=True)\n"
        "os._exit(0)\n"  # skips close(), as a crash would
    )
    (tmp_path / "parent.py").write_text(script)
    output = subprocess.run(
        [sys.executable, str(tmp_path / "parent.py")],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    port = int(output.stdout)
    deadline = time.monotonic() + 30
    while _listening(port=port) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _listening(port=port)


def test_private_server_ignores_osam_serve_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_SERVE_DEVICE", "invalid")
    monkeypatch.setenv("OSAM_MAX_LOADED_MODELS", "0")
    with osam.Client() as client:
        assert client.version() == osam.__version__


def test_private_server_keeps_max_loaded_models() -> None:
    with osam.Client(max_loaded_models=2) as client:
        for model in ["efficientsam:10m", "sam2:tiny"]:
            client.segment(model, _IMAGE, points=[[1280, 800]], point_labels=[1])
        assert [model.name for model in client.ps()] == [
            "sam2:tiny",
            "efficientsam:10m",
        ]
    with pytest.raises(ValueError, match="1 or more"):
        osam.Client(max_loaded_models=0)


def test_failed_start_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "uvicorn.py").write_text("raise ImportError('broken uvicorn')\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    with pytest.raises(RuntimeError, match="failed to start: ImportError: broken"):
        osam.Client().version()
    assert "Traceback" not in capfd.readouterr().err


def test_frozen_application_starts_through_multiprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setting sys.frozen would also make multiprocessing run the child as a
    # frozen executable, which this interpreter is not.
    monkeypatch.setattr(_client, "_start_subprocess", _client._start_frozen)
    with osam.Client() as client:
        assert isinstance(client._process, multiprocessing.process.BaseProcess)
        assert client.version() == osam.__version__
        process = client._process
    assert not process.is_alive()


@contextlib.contextmanager
def _serving() -> Iterator[str]:
    # osam serve in this process, so a test can patch it.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(_server.app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        while not server.started:
            time.sleep(0.01)
        yield f"127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join()


def test_host_checks_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    with _serving() as host:
        monkeypatch.setenv("OSAM_HOST", host)
        assert osam.Client().version() == osam.__version__

        monkeypatch.setattr(_client, "__version__", "0.0.1")
        with pytest.raises(RuntimeError, match="versions differ"):
            osam.Client(host=host)
        with pytest.raises(ValueError, match="http or https"):
            osam.Client(host=f"ftp://{host}")
        with pytest.raises(ValueError, match="device"):
            osam.Client(host=host, device="cpu")
        with pytest.raises(ValueError, match="max_loaded_models"):
            osam.Client(host=host, max_loaded_models=2)


def test_raising_from_progress_stops_the_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled = threading.Event()

    def pull(progress, cancel) -> None:
        for completed in range(1000):
            if cancel.is_set():
                cancelled.set()
                return
            progress("encoder.onnx", completed, 1000)
            time.sleep(0.01)

    model_type = _server._get_model_type(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "pull", pull)

    class Stop(Exception):
        pass

    def progress(line: dict[str, Any]) -> None:
        if line["completed"] == 3:
            raise Stop

    with _serving() as host:
        with pytest.raises(Stop):
            osam.Client(host=host).pull("efficientsam:10m", progress=progress)
        assert cancelled.wait(timeout=5)


def _listening(*, port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return False
    return True


def test_segment_sends_a_repeated_image_by_hash(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded: list[np.ndarray] = []
    encode = _client._json.image_ndarray_to_b64data
    sent: list[set[str]] = []
    request = client._request

    def counting(*, ndarray: np.ndarray) -> str:
        encoded.append(ndarray)
        return encode(ndarray=ndarray)

    def recording(*, method: str, path: str, body: Any = None) -> Any:
        sent.append({"image", "image_hash"} & set(body or {}))
        return request(method=method, path=path, body=body)

    monkeypatch.setattr(_client._json, "image_ndarray_to_b64data", counting)
    monkeypatch.setattr(client, "_request", recording)
    image = _IMAGE[::2, ::2].copy()  # not sent by an earlier test
    for _ in range(2):
        client.segment("efficientsam:10m", image)
    assert sent == [{"image"}, {"image_hash"}]
    assert len(encoded) == 1

    # Another model has no embedding of it yet, so the server asks for it.
    sent.clear()
    client.segment("sam2:tiny", image)
    assert sent == [{"image_hash"}, {"image"}]
    assert len(encoded) == 1

    image[0, 0] += 1  # edited in place, so it is a new image
    client.segment("efficientsam:10m", image)
    assert len(encoded) == 2


def test_timeout_restarts_a_private_server(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = client._process
    send = client._send

    def hang(**kwargs: Any) -> Any:
        monkeypatch.setattr(client, "_send", send)
        raise TimeoutError("timed out")

    monkeypatch.setattr(client, "_send", hang)
    with pytest.raises(TimeoutError):
        client.version()
    assert client._process is not process
    assert client.version() == osam.__version__


def test_timeout_restarts_a_private_server_on_a_retry_or_a_stream(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    send = client._send
    processes: list[Any] = []

    def crash_then_hang(**kwargs: Any) -> Any:
        processes.append(client._process)
        if len(processes) == 1:
            raise ConnectionResetError("crashed")
        monkeypatch.setattr(client, "_send", send)
        raise TimeoutError("timed out")

    monkeypatch.setattr(client, "_send", crash_then_hang)
    with pytest.raises(TimeoutError):
        client.version()
    assert client._process not in processes  # the retry's server restarted too

    class Connection:
        def close(self) -> None:
            pass

    def stall() -> Iterator[bytes]:
        yield b'{"status": "pulling"}'
        raise TimeoutError("timed out")

    process = client._process
    monkeypatch.setattr(
        client, "_open_stream", lambda **kwargs: (Connection(), stall())
    )
    with pytest.raises(TimeoutError):
        client.pull("efficientsam:10m")
    assert client._process is not process


def test_timeout_on_a_silent_host() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:  # accepts, never answers
        port = server.getsockname()[1]
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            osam.Client(host=f"127.0.0.1:{port}", timeout=0.5)
        assert time.monotonic() - start < 5
