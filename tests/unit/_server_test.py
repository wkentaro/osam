import asyncio
import base64
import collections
import json
import pathlib
import threading
import time
import typing
from typing import Final

import anyio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from osam import _json
from osam import _server
from osam import types

_IMAGE_B64: Final = base64.b64encode(
    (
        pathlib.Path(__file__).resolve().parents[2] / "examples/_images/dogs.jpg"
    ).read_bytes()
).decode()
# For requests whose model is replaced, so the image needn't be a real one.
_TINY_IMAGE_B64: Final = _json.image_ndarray_to_b64data(
    ndarray=np.zeros((1, 1, 3), dtype=np.uint8)
)


def _patch_generate(*, monkeypatch: pytest.MonkeyPatch, generate) -> None:
    # Replaces efficientsam:10m's generate, on a server with no model loaded.
    monkeypatch.setattr(_server, "_loaded", collections.OrderedDict())
    monkeypatch.setattr(_server, "_embeddings", collections.OrderedDict())
    model_type = _server._apis.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "generate", generate)


def test_openapi_documents_every_route() -> None:
    response = TestClient(_server.app).get("/openapi.json")
    assert response.status_code == 200, response.text
    assert {
        "/api/generate",
        "/api/models",
        "/api/pull",
        "/api/models/{name}",
        "/api/ps",
        "/api/version",
    } <= set(response.json()["paths"])


def test_list_models() -> None:
    response = TestClient(_server.app).get("/api/models")
    models = {model["name"]: model for model in response.json()["models"]}
    assert models["efficientsam:10m"]["size"] > 0
    assert models["efficientsam:10m"]["metadata"]["license_spdx"] == "Apache-2.0"


def test_pull_streams_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    model_type = _server._apis.get_model_type_by_name(name="efficientsam:10m")

    def pull(progress, cancel):
        progress("encoder.onnx", 1, 2)
        progress("encoder.onnx", 2, 2)

    monkeypatch.setattr(model_type, "pull", pull)
    client = TestClient(_server.app)
    response = client.post("/api/pull", json={"model": "efficientsam:10m"})
    assert response.headers["content-type"] == "application/x-ndjson"
    assert [json.loads(line) for line in response.text.splitlines()] == [
        {"status": "pulling encoder.onnx", "completed": 1, "total": 2},
        {"status": "pulling encoder.onnx", "completed": 2, "total": 2},
        {"status": "success"},
    ]

    def fail(progress, cancel):
        raise RuntimeError("disk full")

    monkeypatch.setattr(model_type, "pull", fail)
    response = client.post("/api/pull", json={"model": "efficientsam:10m"})
    assert response.text.splitlines()[-1] == json.dumps({"error": "disk full"})
    response = client.post("/api/pull", json={"model": "missing"})
    assert response.status_code == 404


def test_generate_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_server.app)
    response = client.post("/api/generate", json={"model": "missing"})
    assert response.status_code == 404
    assert "not found" in response.json()["error"]
    truncated: str = base64.b64encode(base64.b64decode(_IMAGE_B64)[:1000]).decode()
    for body in [
        {"image": "eA=="},
        {"image": truncated},
        {"annotations": [{"mask": "eA=="}]},
    ]:
        response = client.post(
            "/api/generate", json={"model": "efficientsam:10m", **body}
        )
        assert response.status_code == 422, body
        assert "not an image file" in response.json()["error"]

    def fail(self, request):
        assert self.device == "cpu"
        raise RuntimeError("out of GPU memory")

    monkeypatch.setattr(_server, "_DEVICE", "cpu")
    _patch_generate(monkeypatch=monkeypatch, generate=fail)
    response = TestClient(_server.app, raise_server_exceptions=False).post(
        "/api/generate", json={"model": "efficientsam:10m", "image": _TINY_IMAGE_B64}
    )
    assert response.status_code == 500
    assert response.json() == {"error": "out of GPU memory"}


def test_queued_generates_leave_the_thread_pool_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(self, request):
        release.wait()
        raise ValueError("released")

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # More than the thread pool's 40 workers.
            body = {"model": "efficientsam:10m", "image": _TINY_IMAGE_B64}
            tasks = [
                asyncio.create_task(client.post("/api/generate", json=body))
                for _ in range(50)
            ]
            await asyncio.sleep(0.2)
            try:
                with anyio.fail_after(5):
                    # index is sync, so it needs a free worker.
                    assert (await client.get("/")).status_code == 200
            finally:
                release.set()
            responses = await asyncio.gather(*tasks)
        assert {response.status_code for response in responses} == {400}

    _patch_generate(monkeypatch=monkeypatch, generate=blocked)
    # Twice, on two event loops, as separate app lifecycles would.
    for _ in range(2):
        release = threading.Event()
        asyncio.run(scenario())


def test_generate_keeps_image_embedding_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = types.ImageEmbedding(
        original_height=1,
        original_width=1,
        embedding=np.zeros((1, 1, 1), dtype=np.float32),
    )
    _patch_generate(
        monkeypatch=monkeypatch,
        generate=lambda self, request: types.GenerateResponse(
            model=request.model, image_embedding=embedding, annotations=[]
        ),
    )
    client = TestClient(_server.app)

    response = client.post(
        "/api/generate", json={"model": "efficientsam:10m", "image": _TINY_IMAGE_B64}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"model": "efficientsam:10m", "annotations": []}

    response = client.post(
        "/api/generate",
        json={
            "model": "efficientsam:10m",
            "image_embedding": embedding.model_dump(),
        },
    )
    assert response.status_code == 422
    assert "image_embedding" in response.json()["error"]


def test_generate_reuses_cached_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    _server._embeddings.clear()
    client = TestClient(_server.app)
    prompt = {"points": [[1280, 800]], "point_labels": [1]}
    body = {"model": "efficientsam:10m", "image": _IMAGE_B64, "prompt": prompt}
    first = client.post("/api/generate", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["annotations"][0]["mask"]

    def fail(*args, **kwargs):
        raise AssertionError("encoded the image again")

    model = _server._loaded["efficientsam:10m"].model
    monkeypatch.setattr(type(model), "encode_image", fail)
    second = client.post("/api/generate", json=body)
    assert second.status_code == 200, second.text
    assert second.json()["annotations"] == first.json()["annotations"]


def test_routing_errors_use_the_error_shape() -> None:
    client = TestClient(_server.app)
    assert client.get("/api/missing").json() == {"error": "Not Found"}
    response = client.post("/api/version")
    assert response.status_code == 405
    assert response.json() == {"error": "Method Not Allowed"}
    assert response.headers["allow"] == "GET"


def test_version() -> None:
    response = TestClient(_server.app).get("/api/version")
    assert response.json() == {"version": _server.__version__}


@pytest.mark.parametrize(
    ("value", "seconds"),
    [(300, 300), ("10", 10), ("5m", 300), ("1h30m", 5400), ("-1s", -1)],
)
def test_parse_duration(value, seconds) -> None:
    assert _server._parse_duration(value) == seconds


def test_keep_alive_unloads_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_server.app)
    body = {"model": "efficientsam:10m", "image": _IMAGE_B64}

    response = client.post("/api/generate", json={**body, "keep_alive": -1})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert running["name"] == "efficientsam:10m"
    assert running["device"] in typing.get_args(types.Device)
    assert running["expires_at"] is None

    response = client.post("/api/generate", json={**body, "keep_alive": "0.2s"})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert running["expires_at"] > time.time()
    deadline = time.monotonic() + 5
    while client.get("/api/ps").json()["models"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client.get("/api/ps").json() == {"models": []}
    assert not _server._embeddings

    # A request with only a model sets its keep_alive, so 0 unloads it. It
    # loads no model, and leaves another loaded model alone.
    unload = {"model": "efficientsam:10m", "keep_alive": 0}
    response = client.post("/api/generate", json={"model": "efficientsam:10m"})
    assert response.json() == {"model": "efficientsam:10m", "annotations": []}
    assert client.get("/api/ps").json() == {"models": []}
    client.post("/api/generate", json={**body, "keep_alive": -1})
    assert client.post("/api/generate", json={**unload, "model": "sam2:tiny"}).json()
    assert client.get("/api/ps").json()["models"]
    assert client.post("/api/generate", json=unload).status_code == 200
    assert client.get("/api/ps").json() == {"models": []}
    prompt = {"points": [[1, 1]], "point_labels": [1]}
    response = client.post("/api/generate", json={**unload, "prompt": prompt})
    assert response.status_code == 400

    for keep_alive in ["1x", "inf", 1e20, None, []]:
        response = client.post("/api/generate", json={**body, "keep_alive": keep_alive})
        assert response.status_code == 422, keep_alive


def test_two_models_stay_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_server.app)

    def generate(model: str) -> None:
        body = {"model": model, "image": _TINY_IMAGE_B64, "keep_alive": -1}
        response = client.post("/api/generate", json=body)
        assert response.status_code == 200, response.text

    def running() -> list[str]:
        return [model["name"] for model in client.get("/api/ps").json()["models"]]

    monkeypatch.setattr(_server, "_loaded", collections.OrderedDict())
    monkeypatch.setattr(_server, "_embeddings", collections.OrderedDict())
    generate("efficientsam:10m")
    generate("sam2:tiny")
    assert running() == ["sam2:tiny", "efficientsam:10m"]
    sam2 = _server._loaded["sam2:tiny"].model
    generate("efficientsam:10m")
    assert _server._loaded["sam2:tiny"].model is sam2  # not reloaded
    assert running() == ["efficientsam:10m", "sam2:tiny"]

    # A full server unloads the least recently used model and its embeddings.
    client.post("/api/generate", json={"model": "sam2:tiny", "keep_alive": 0})
    assert [model for model, _ in _server._embeddings] == ["efficientsam:10m"]
    monkeypatch.setattr(_server, "_MAX_LOADED_MODELS", 1)
    generate("sam2:tiny")
    assert running() == ["sam2:tiny"]
    assert [model for model, _ in _server._embeddings] == ["sam2:tiny"]


def test_failed_load_unloads_the_others_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_server, "_loaded", collections.OrderedDict())
    monkeypatch.setattr(_server, "_embeddings", collections.OrderedDict())
    client = TestClient(_server.app)
    body = {"image": _TINY_IMAGE_B64, "keep_alive": -1}
    response = client.post("/api/generate", json={**body, "model": "efficientsam:10m"})
    assert response.status_code == 200, response.text
    sam2 = _server._apis.get_model_type_by_name(name="sam2:tiny")

    def offline(*args, **kwargs) -> None:
        raise OSError("network is unreachable")

    # A failed download unloads nothing.
    with monkeypatch.context() as patch:
        patch.setattr(sam2, "pull", offline)
        response = TestClient(_server.app, raise_server_exceptions=False).post(
            "/api/generate", json={**body, "model": "sam2:tiny"}
        )
    assert response.json() == {"error": "network is unreachable"}
    assert list(_server._loaded) == ["efficientsam:10m"]

    loaded_at_each_load: list[list[str]] = []

    init = sam2.__init__

    def out_of_memory_once(self, *args, **kwargs) -> None:
        loaded_at_each_load.append(list(_server._loaded))
        if len(loaded_at_each_load) == 1:
            raise RuntimeError("out of memory")
        init(self, *args, **kwargs)

    monkeypatch.setattr(sam2, "__init__", out_of_memory_once)
    response = client.post("/api/generate", json={**body, "model": "sam2:tiny"})
    assert response.status_code == 200, response.text
    assert loaded_at_each_load == [["efficientsam:10m"], []]
    assert list(_server._loaded) == ["sam2:tiny"]
