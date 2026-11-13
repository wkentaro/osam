import contextlib
import http.client
import http.server
import json
import logging
import multiprocessing
import pathlib
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import Final

import numpy as np
import onnxruntime
import PIL.Image
import pytest
import uvicorn

import osam
from osam import _errors
from osam import _http
from osam import _registry
from osam import _runtime
from osam import _server

_IMAGE: Final = np.asarray(
    PIL.Image.open(
        pathlib.Path(__file__).resolve().parents[2] / "examples/_images/dogs.jpg"
    )
)


def _backend(client: osam.Client) -> _http.HttpBackend:
    assert isinstance(client._backend, _http.HttpBackend)
    return client._backend


@pytest.fixture(scope="module")
def client():
    with osam.Client() as client:
        yield client


@pytest.fixture(scope="module", params=["private server", "in process"])
def any_client(request: pytest.FixtureRequest, client: osam.Client):
    if request.param == "private server":
        yield client
        return
    with osam.Client(in_process=True) as in_process:
        yield in_process


def test_client_segments(any_client: osam.Client) -> None:
    client = any_client
    assert client.version() == osam.__version__
    assert "efficientsam:10m" in {model.name for model in client.list()}

    (annotation,) = client.segment(
        "efficientsam:10m", _IMAGE, points=np.array([[1280, 800]]), point_labels=[1]
    )
    assert annotation.mask is not None
    assert annotation.mask.any()
    (unmasked,) = client.segment(
        "efficientsam:10m",
        _IMAGE,
        points=np.array([[1280, 800]]),
        point_labels=[1],
        include_masks=False,
    )
    assert unmasked.mask is None
    assert unmasked.bounding_box == annotation.bounding_box
    # The same on either backend.
    with osam.Client(in_process=True) as in_process:
        (local,) = in_process.segment(
            "efficientsam:10m", _IMAGE, points=np.array([[1280, 800]]), point_labels=[1]
        )
    assert local.bounding_box == annotation.bounding_box
    assert local.mask is not None
    assert (local.mask == annotation.mask).all()
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


def test_unknown_model_raises_lookup_error(any_client: osam.Client) -> None:
    calls: list[Callable[[], object]] = [
        lambda: any_client.segment("missing", _IMAGE),
        lambda: any_client.load("missing"),
        lambda: any_client.pull("missing"),
        lambda: any_client.remove("missing"),
    ]
    for call in calls:
        with pytest.raises(LookupError, match="not found") as e:
            call()
        if isinstance(e.value, osam.ResponseError):
            assert (e.value.status_code, e.value.code) == (404, "model_not_found")


def test_invalid_arguments_raise_before_a_request(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(**kwargs: Any) -> Any:
        raise AssertionError("sent a request")

    monkeypatch.setattr(_backend(client), "_request", unexpected)
    with pytest.raises(ValueError, match="same length"):
        client.segment("efficientsam:10m", _IMAGE, points=[[1, 2]], point_labels=[])
    with pytest.raises(ValueError, match="duration"):
        client.load("efficientsam:10m", keep_alive="soon")


def test_in_process_takes_no_server_settings() -> None:
    with pytest.raises(ValueError, match="host"):
        osam.Client(host="127.0.0.1:11368", in_process=True)
    with pytest.raises(ValueError, match="timeout"):
        osam.Client(timeout=600, in_process=True)
    with pytest.raises(ValueError, match="max_loaded_models"):
        osam.Client(max_loaded_models=0, in_process=True)


def test_in_process_close_unloads_its_models() -> None:
    with osam.Client(in_process=True, device="cpu") as client:
        client.load("efficientsam:10m", keep_alive=-1)
        assert [(model.name, model.device) for model in client.ps()] == [
            ("efficientsam:10m", "cpu")
        ]
    assert client.ps() == []


def test_private_server_needs_its_token(client: osam.Client) -> None:
    connection = _http.http.client.HTTPConnection(*_backend(client)._endpoint[:2])
    connection.request("GET", "/api/version")
    assert connection.getresponse().status == 401


def test_private_server_restarts_after_a_crash(client: osam.Client) -> None:
    for call in [client.version, lambda: client.pull("efficientsam:10m")]:
        process = _backend(client)._process
        assert process is not None
        process.kill()
        _http._wait(process=process, timeout=None)
        call()


def test_concurrent_requests_survive_one_restart(client: osam.Client) -> None:
    process = _backend(client)._process
    assert process is not None
    process.kill()
    _http._wait(process=process, timeout=None)
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
        "print(client._backend._endpoint[1], flush=True)\n"
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


def test_unavailable_device_fails_at_start(capfd: pytest.CaptureFixture[str]) -> None:
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():  # type: ignore[possibly-missing-attribute]
        pytest.skip("CUDA is available")
    kwargs_list: list[dict[str, Any]] = [{}, {"in_process": True}]
    for kwargs in kwargs_list:
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
            osam.Client(device="cuda", **kwargs)
    assert "Traceback" not in capfd.readouterr().err


@pytest.mark.skipif(sys.platform == "win32", reason="closes fd 2 with sh")
def test_private_server_starts_without_a_stderr() -> None:
    # As under pythonw.exe, where the child starts with sys.stderr None.
    result = subprocess.run(
        [
            "sh",
            "-c",
            'exec 2>&-; exec "$0" -c "from osam._http import _serve_stdio; '
            '_serve_stdio()"',
            sys.executable,
        ],
        input=b'{"token": "t", "settings": {}}\n',
        stdout=subprocess.PIPE,
        timeout=60,
    )
    assert result.returncode == 0
    assert b'"port"' in result.stdout


def test_frozen_application_starts_through_multiprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setting sys.frozen would also make multiprocessing run the child as a
    # frozen executable, which this interpreter is not.
    monkeypatch.setattr(_http, "_start_subprocess", _http._start_frozen)
    with osam.Client() as client:
        process = _backend(client)._process
        assert isinstance(process, multiprocessing.process.BaseProcess)
        assert client.version() == osam.__version__
    assert not process.is_alive()


@contextlib.contextmanager
def _serving(*, runtime: _runtime.Runtime | None = None) -> Iterator[str]:
    # osam serve in this process, so a test can patch it.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    runtime = runtime or _runtime.Runtime()
    server = uvicorn.Server(
        uvicorn.Config(_server.create_app(runtime=runtime), log_level="warning")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        while not server.started:
            time.sleep(0.01)
        yield f"127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join()
        runtime.close()


def test_host_checks_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    with _serving() as host:
        monkeypatch.setenv("OSAM_HOST", host)
        assert osam.Client().version() == osam.__version__

        monkeypatch.setattr(_http, "__version__", "0.0.1")
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

    def pull(device, progress, cancel) -> None:
        for completed in range(1000):
            if cancel.is_set():
                cancelled.set()
                return
            progress("encoder.onnx", completed, 1000)
            time.sleep(0.01)

    model_type = _registry.get_model_type_by_name(name="efficientsam:10m")
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


def test_cancel_stops_a_stalled_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    # A download that sends one chunk and then nothing, until it is cancelled.
    stopped = threading.Event()

    def pull(device, progress, cancel) -> None:
        progress("encoder.onnx", 1, 1000)
        assert cancel.wait(timeout=10)
        stopped.set()
        raise osam.types.PullCancelledError("cancelled")

    model_type = _registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "pull", pull)

    with _serving() as host, osam.Client(in_process=True) as in_process:
        for client in [osam.Client(host=host), in_process]:
            stopped.clear()
            cancel = threading.Event()
            started = time.monotonic()
            with pytest.raises(osam.types.PullCancelledError):
                client.pull(
                    "efficientsam:10m",
                    progress=lambda _: threading.Timer(0.2, cancel.set).start(),
                    cancel=cancel,
                )
            assert time.monotonic() - started < 2
            assert stopped.wait(timeout=5)


@pytest.mark.parametrize(
    "sent",
    [
        pytest.param((200, b'{"status": "pulling encoder.onnx"}\n'), id="after a line"),
        pytest.param((200, b'{"status": "pull'), id="mid line"),
        pytest.param((502, b'{"error": "fail'), id="mid an error"),
        pytest.param(None, id="before the headers"),
    ],
)
def test_cancel_stops_a_stalled_pull_on_a_closing_connection(
    sent: tuple[int, bytes] | None,
) -> None:
    # A server, such as one behind a proxy, that answers "Connection: close",
    # so http.client hands the socket to the response, and stalls after
    # sending what it sent.
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._send(
                status=200, body=json.dumps({"version": osam.__version__}).encode()
            )

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            if sent is not None:
                self._send(status=sent[0], body=sent[1])
            release.wait(timeout=10)

        def _send(self, *, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, *args: object) -> None:
            pass

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            client = osam.Client(host=f"127.0.0.1:{server.server_port}")
            cancel = threading.Event()
            threading.Timer(0.3, cancel.set).start()
            started = time.monotonic()
            with pytest.raises(osam.types.PullCancelledError):
                client.pull("efficientsam:10m", cancel=cancel)
            assert time.monotonic() - started < 2
        finally:
            release.set()
            server.shutdown()
            thread.join()


def test_failed_pull_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def pull(device, progress, cancel) -> None:
        raise RuntimeError("Failed to download encoder.onnx")

    model_type = _registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "pull", pull)

    with _serving() as host, osam.Client(in_process=True) as in_process:
        for client in [osam.Client(host=host), in_process]:
            with pytest.raises(RuntimeError, match="Failed to download"):
                client.pull("efficientsam:10m")


def test_failed_download_before_a_load_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def pull(*args, **kwargs) -> None:
        raise _errors.DownloadError("Failed to download encoder.onnx")

    model_type = _registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "pull", pull)

    with _serving() as host, osam.Client(in_process=True) as in_process:
        for client in [osam.Client(host=host), in_process]:
            for call in [
                lambda: client.segment("efficientsam:10m", _IMAGE),
                lambda: client.load("efficientsam:10m"),
            ]:
                with pytest.raises(RuntimeError, match="Failed to download") as e:
                    call()
                if isinstance(e.value, osam.ResponseError):
                    assert (e.value.status_code, e.value.code) == (502, "pull_failed")


def test_failed_download_does_not_restart_a_private_server(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _backend(client)._process
    sent: list[str] = []

    def fail(*, path: str, **kwargs: Any) -> Any:
        sent.append(path)
        raise _http._response_error(
            status_code=502, data=b'{"error": "Failed to download encoder.onnx"}'
        )

    monkeypatch.setattr(_backend(client), "_send", fail)
    with pytest.raises(RuntimeError, match="Failed to download"):
        client.load("efficientsam:10m")
    assert sent == ["/api/load"]  # not sent again
    assert _backend(client)._process is process


def _listening(*, port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return False
    return True


def test_segment_sends_a_repeated_image_by_reference(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded: list[np.ndarray] = []
    encode = _http._json.image_ndarray_to_b64data
    sent: list[set[str]] = []
    request = _backend(client)._request

    def counting(*, ndarray: np.ndarray) -> str:
        encoded.append(ndarray)
        return encode(ndarray=ndarray)

    def recording(*, method: str, path: str, body: Any = None) -> Any:
        image: str = (body or {}).get("image", "")
        sent.append({"ref" if image.startswith("sha256:") else "image"})
        return request(method=method, path=path, body=body)

    monkeypatch.setattr(_http._json, "image_ndarray_to_b64data", counting)
    monkeypatch.setattr(_backend(client), "_request", recording)
    image = _IMAGE[::2, ::2].copy()  # not sent by an earlier test
    for _ in range(2):
        client.segment("efficientsam:10m", image)
    assert sent == [{"image"}, {"ref"}]
    assert len(encoded) == 1

    # Another model has no embedding of it yet, so the server asks for it.
    sent.clear()
    client.segment("sam2:tiny", image)
    assert sent == [{"ref"}, {"image"}]
    assert len(encoded) == 1

    image[0, 0] += 1  # edited in place, so it is a new image
    client.segment("efficientsam:10m", image)
    assert len(encoded) == 2


def test_timeout_restarts_a_private_server(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = _backend(client)._process
    send = _backend(client)._send

    def hang(**kwargs: Any) -> Any:
        monkeypatch.setattr(_backend(client), "_send", send)
        raise TimeoutError("timed out")

    monkeypatch.setattr(_backend(client), "_send", hang)
    with pytest.raises(TimeoutError):
        client.version()
    assert _backend(client)._process is not process
    assert client.version() == osam.__version__


def test_timeout_restarts_a_private_server_on_a_retry_or_a_stream(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    send = _backend(client)._send
    processes: list[Any] = []

    def crash_then_hang(**kwargs: Any) -> Any:
        processes.append(_backend(client)._process)
        if len(processes) == 1:
            raise ConnectionResetError("crashed")
        monkeypatch.setattr(_backend(client), "_send", send)
        raise TimeoutError("timed out")

    monkeypatch.setattr(_backend(client), "_send", crash_then_hang)
    with pytest.raises(TimeoutError):
        client.version()
    assert (
        _backend(client)._process not in processes
    )  # the retry's server restarted too

    class Connection:
        def close(self) -> None:
            pass

    def stall() -> Iterator[bytes]:
        yield b'{"status": "pulling"}'
        raise TimeoutError("timed out")

    process = _backend(client)._process
    monkeypatch.setattr(
        _backend(client), "_open_stream", lambda **kwargs: (Connection(), stall())
    )
    with pytest.raises(TimeoutError):
        client.pull("efficientsam:10m")
    assert _backend(client)._process is not process


def test_timeout_on_a_silent_host() -> None:
    with socket.create_server(("127.0.0.1", 0)) as server:  # accepts, never answers
        port = server.getsockname()[1]
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            osam.Client(host=f"127.0.0.1:{port}", timeout=0.5)
        assert time.monotonic() - start < 5


def test_track_sends_frames_the_server_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit._server_test import _patch_tracking

    applied: list[str] = _patch_tracking(
        monkeypatch=monkeypatch,
        model_type=_registry.get_model_type_by_name(name="efficientsam:10m"),
    )
    frames = [np.full((2, 2, 3), value, np.uint8) for value in range(3)]
    prompt = {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}
    with _serving() as host, osam.Client(in_process=True) as in_process:
        client = osam.Client(host=host)
        backend = _backend(client)
        posted: list[list[bool]] = []
        stream = backend._stream

        def recording(*, path: str, body: dict[str, Any]):
            posted.append([f.startswith("sha256:") for f in body["frames"]])
            return stream(path=path, body=body)

        monkeypatch.setattr(backend, "_stream", recording)
        events: list[osam.types.TrackProgress] = []
        for any_client in [client, in_process]:
            response = any_client.track(
                "efficientsam:10m", frames[:2], [prompt], progress=events.append
            )
            assert [f.frame for f in response.frames] == [0, 1]
            assert response.frames[1].annotations[0].object_id == 0
            assert response.done.cache == "miss"
        # By reference, then the two frames it lacked inline.
        assert posted == [[True, True], [False, False]]
        assert events[0].frame is None
        assert any(event.frame == 1 for event in events)

        posted.clear()
        add = {"type": "add", "frame": 2, "box": [0, 0, 1, 1]}
        response = client.track(
            "efficientsam:10m",
            frames,
            [prompt, {**add, "base": response.done.result}],
            include_masks=False,
        )
        assert posted == [[True, True, True], [True, True, False]]
        assert response.done.cache == "miss"  # three frames, another video
        assert response.frames[2].annotations[1].mask is None
        assert applied.count("add") == 1

        with pytest.raises(osam.ResponseError) as e:
            client.track("efficientsam:10m", frames, [prompt, {**add, "base": "x"}])
        assert e.value.code == "base_changed"
        assert e.value.status_code == 409
        assert e.value.result == f"sha256:{1:064d}"


def test_track_sends_every_frame_when_the_server_loses_more(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As after a restart between the first 412 and the frames it asked for.
    frames = [np.full((2, 2, 3), value, np.uint8) for value in range(3)]
    refs = [_http._json.image_ref(image=frame) for frame in frames]
    posted: list[list[bool]] = []
    done = {
        "done": True,
        "objects": [],
        "result": "sha256:0",
        "cache": "miss",
        "images": refs,
    }

    def stream(*, path: str, body: dict[str, Any]):
        posted.append([f.startswith("sha256:") for f in body["frames"]])
        if len(posted) < 3:
            missing = refs[2:] if len(posted) == 1 else refs[:2]
            raise osam.ResponseError("", 412, "missing_images", images=missing)
        yield done

    monkeypatch.setattr(_backend(client), "_stream", stream)
    prompt = {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}
    assert client.track("efficientsam:10m", frames, [prompt]).done.result == "sha256:0"
    assert posted == [[True, True, True], [True, True, False], [False, False, False]]


@pytest.mark.parametrize("cancel_from_callback", [False, True])
def test_disconnected_tracking_stops_before_the_clip_finishes(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cancel_from_callback: bool,
) -> None:
    from . import _fakes
    from ._server_test import _FakeTimeline
    from ._server_test import _patch_tracking

    model = _fakes.model_type(name="fake:latest")
    runtime = _runtime.Runtime(model_types=[model])
    _patch_tracking(monkeypatch=monkeypatch, model_type=model)
    started, stopped = threading.Event(), threading.Event()
    processed: list[int] = []

    def slow(self, *, action, on_frame=None):
        self._count += 1
        started.set()
        try:
            for frame in range(self._num_frames):
                if on_frame is not None:
                    on_frame(frame)
                processed.append(frame)
                time.sleep(0.01)
        finally:
            stopped.set()

    monkeypatch.setattr(_FakeTimeline, "apply", slow)
    frames = [np.zeros((1, 1, 3), np.uint8)] * 200
    prompt = {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}
    with _serving(runtime=runtime) as host, osam.Client(host=host) as client:
        logger = logging.getLogger("uvicorn.error")
        monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
        if cancel_from_callback:

            def progress(event: osam.types.TrackProgress) -> None:
                if event.frame == 3:
                    raise RuntimeError("cancel from UI")

            with pytest.raises(RuntimeError, match="cancel from UI"):
                client.track("fake", frames, [prompt], progress=progress)
        else:
            connection = http.client.HTTPConnection(host, timeout=5)
            inline = _http._json.image_ndarray_to_b64data(ndarray=frames[0])
            connection.request(
                "POST",
                "/api/track",
                json.dumps(
                    {"model": "fake", "frames": [inline] * 200, "actions": [prompt]}
                ),
                {"Content-Type": "application/json"},
            )
            assert started.wait(5)
            connection.close()
        assert stopped.wait(5)
        assert len(processed) < len(frames)
        # No stale partial timeline, and the worker is available again.
        client.load("fake", keep_alive=-1)
        assert runtime._timeline is None
        assert "Video tracking failed" not in caplog.text
        response = client.track(
            "fake", frames[:1], [prompt], progress=lambda event: None
        )
        assert response.done.cache == "miss"
        assert response.frames[0].frame == 0


def test_cancelled_queued_tracking_does_not_run_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from . import _fakes
    from ._server_test import _FakeTimeline
    from ._server_test import _patch_tracking

    model = _fakes.model_type(name="fake:latest")
    runtime = _runtime.Runtime(model_types=[model])
    _patch_tracking(monkeypatch=monkeypatch, model_type=model)
    started, release = threading.Event(), threading.Event()
    calls: list[int] = []
    prompt = {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}

    def slow(self, *, action, on_frame=None):
        calls.append(action.frame)
        self._count += 1
        started.set()
        assert release.wait(5)

    monkeypatch.setattr(_FakeTimeline, "apply", slow)
    errors: list[BaseException] = []
    with _serving(runtime=runtime) as host, osam.Client(host=host) as client:

        def first() -> None:
            try:
                client.track(
                    "fake",
                    [np.zeros((1, 1, 3), np.uint8)],
                    [prompt],
                )
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=first)
        thread.start()
        try:
            assert started.wait(5)
            connection = http.client.HTTPConnection(host, timeout=5)
            inline = _http._json.image_ndarray_to_b64data(
                ndarray=np.ones((1, 1, 3), np.uint8)
            )
            connection.request(
                "POST",
                "/api/track",
                json.dumps(
                    {
                        "model": "fake",
                        "frames": [inline],
                        "actions": [prompt],
                    }
                ),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            assert json.loads(response.readline())["progress"] is True
            response.close()
            connection.close()
            time.sleep(0.2)  # let the server receive the disconnect before unblocking
        finally:
            release.set()
            thread.join(5)
        assert not thread.is_alive()
        assert not errors
        # This queued load also proves the cancelled tracking worker has exited.
        client.load("fake", keep_alive=-1)
        assert calls == [0]


def test_track_stream_failure_closes_connection_before_private_restart(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _backend(client)
    closed: list[bool] = []
    restarts: list[object] = []

    def stream(**kwargs):
        try:
            yield {
                "error": "model failed",
                "code": "internal_error",
                "status_code": 500,
            }
        finally:
            closed.append(True)

    def restart(*, process, error):
        assert closed == [True]
        restarts.append(process)

    monkeypatch.setattr(backend, "_stream", stream)
    monkeypatch.setattr(backend, "_restart", restart)
    with pytest.raises(osam.ResponseError) as exc:
        client.track(
            "sam3",
            [np.zeros((1, 1, 3), np.uint8)],
            [{"type": "prompt", "frame": 0, "texts": ["person"]}],
            progress=lambda event: None,
        )
    assert exc.value.status_code == 500
    assert restarts == [backend._process]


def test_track_never_retries_an_exception_from_the_progress_callback(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed: list[bool] = []

    def stream(**kwargs):
        try:
            yield {"progress": True, "total_frames": 1}
        finally:
            closed.append(True)

    error = osam.ResponseError("cancel", 412, "missing_images")

    def progress(event: osam.types.TrackProgress) -> None:
        raise error

    monkeypatch.setattr(_backend(client), "_stream", stream)
    with pytest.raises(osam.ResponseError) as exc:
        client.track(
            "sam3",
            [np.zeros((1, 1, 3), np.uint8)],
            [{"type": "prompt", "frame": 0, "texts": ["person"]}],
            progress=progress,
        )
    assert exc.value is error
    assert closed == [True]


@pytest.mark.parametrize("done", [False, True])
def test_track_requires_a_terminal_record_and_closes_at_done(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch, done: bool
) -> None:
    closed: list[bool] = []

    def stream(**kwargs):
        try:
            yield {"frame": 0, "annotations": []}
            if done:
                yield {
                    "done": True,
                    "objects": [],
                    "result": "sha256:0",
                    "cache": "hit",
                    "images": [],
                }
                raise AssertionError("read past the terminal record")
        finally:
            closed.append(True)

    monkeypatch.setattr(_backend(client), "_stream", stream)
    args = (
        "sam3",
        [np.zeros((1, 1, 3), np.uint8)],
        [{"type": "prompt", "frame": 0, "texts": ["person"]}],
    )
    if done:
        assert client.track(*args).done.result == "sha256:0"
    else:
        with pytest.raises(RuntimeError, match="without a done record"):
            client.track(*args)
    assert closed == [True]
