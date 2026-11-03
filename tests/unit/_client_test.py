import multiprocessing
import pathlib
import socket
import subprocess
import sys
import threading
import time
from typing import Final

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


def test_private_server_generates(client: osam.Client) -> None:
    assert client.version() == osam.__version__
    assert "efficientsam:10m" in {model.name for model in client.list()}

    response = client.generate(
        "efficientsam:10m",
        image=_IMAGE,
        prompt=osam.types.Prompt(
            points=np.array([[1280, 800]]), point_labels=np.array([1])
        ),
    )
    assert response.annotations[0].mask is not None
    assert response.annotations[0].mask.any()
    assert [model.name for model in client.ps()] == ["efficientsam:10m"]

    with pytest.raises(osam.ResponseError) as e:
        client.generate("missing")
    assert e.value.status_code == 404
    assert "not found" in str(e.value)


def test_private_server_needs_its_token(client: osam.Client) -> None:
    connection = _client.http.client.HTTPConnection(client._host, client._port)
    connection.request("GET", "/api/version")
    assert connection.getresponse().status == 401


def test_private_server_restarts_after_a_crash(client: osam.Client) -> None:
    assert client._process is not None
    for call in [client.version, lambda: client.pull("efficientsam:10m")]:
        assert client._process is not None
        client._process.kill()
        _client._wait(process=client._process, timeout=None)
        call()


def test_private_server_exits_with_its_parent(tmp_path: pathlib.Path) -> None:
    # No `if __name__ == "__main__"`, as in a quick script.
    script = (
        "import os, osam\n"
        "client = osam.Client()\n"
        "print(client._port, flush=True)\n"
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


def test_host_checks_the_version(monkeypatch: pytest.MonkeyPatch) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(_server.app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    thread.start()
    try:
        while not server.started:
            time.sleep(0.01)
        host = f"127.0.0.1:{sock.getsockname()[1]}"

        monkeypatch.setenv("OSAM_HOST", host)
        assert osam.Client().version() == osam.__version__

        monkeypatch.setattr(_client, "__version__", "0.0.1")
        with pytest.raises(RuntimeError, match="versions differ"):
            osam.Client(host=host)
        with pytest.raises(ValueError, match="device"):
            osam.Client(host=host, device="cpu")
    finally:
        server.should_exit = True
        thread.join()


def _listening(*, port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
    except OSError:
        return False
    return True


def test_generate_encodes_a_repeated_image_once(
    client: osam.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoded: list[np.ndarray] = []
    encode = _client._json.image_ndarray_to_b64data

    def counting(*, ndarray: np.ndarray) -> str:
        encoded.append(ndarray)
        return encode(ndarray=ndarray)

    monkeypatch.setattr(_client._json, "image_ndarray_to_b64data", counting)
    image = _IMAGE[::2, ::2].copy()  # not sent by an earlier test
    for _ in range(2):
        client.generate("efficientsam:10m", image=image)
    assert len(encoded) == 1
    image[0, 0] += 1  # edited in place, so it is a new image
    client.generate("efficientsam:10m", image=image)
    assert len(encoded) == 2
