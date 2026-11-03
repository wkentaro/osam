import asyncio
import base64
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

from osam import _server
from osam import apis
from osam import types

_IMAGE_B64: Final = base64.b64encode(
    (
        pathlib.Path(__file__).resolve().parents[2] / "examples/_images/dogs.jpg"
    ).read_bytes()
).decode()


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
    model_type = _server.apis.get_model_type_by_name(name="efficientsam:10m")

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
    response = client.post("/api/generate", json={"model": "efficientsam:10m"})
    assert response.status_code == 400

    def fail(request, device):
        assert device == "cpu"
        raise RuntimeError("out of GPU memory")

    monkeypatch.setattr(_server, "_DEVICE", "cpu")

    monkeypatch.setattr(_server.apis, "generate", fail)
    response = client.post("/api/generate", json={"model": "efficientsam:10m"})
    assert response.status_code == 500
    assert response.json() == {"error": "out of GPU memory"}


def test_queued_generates_leave_the_thread_pool_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(request, device):
        release.wait()
        raise ValueError("released")

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=_server.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # More than the thread pool's 40 workers.
            body = {"model": "efficientsam:10m"}
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

    monkeypatch.setattr(_server.apis, "generate", blocked)
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
    monkeypatch.setattr(
        _server.apis,
        "generate",
        lambda request, **kwargs: types.GenerateResponse(
            model=request.model, image_embedding=embedding, annotations=[]
        ),
    )
    client = TestClient(_server.app)

    response = client.post("/api/generate", json={"model": "efficientsam:10m"})
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

    monkeypatch.setattr(type(apis.running_model), "encode_image", fail)
    second = client.post("/api/generate", json=body)
    assert second.status_code == 200, second.text
    assert second.json()["annotations"] == first.json()["annotations"]


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

    for keep_alive in ["1x", "inf", None, []]:
        response = client.post("/api/generate", json={**body, "keep_alive": keep_alive})
        assert response.status_code == 422, keep_alive
