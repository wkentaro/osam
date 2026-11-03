import asyncio
import json
import threading

import anyio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from osam import _server
from osam import types


def test_openapi_documents_every_route() -> None:
    response = TestClient(_server.app).get("/openapi.json")
    assert response.status_code == 200, response.text
    assert {
        "/api/generate",
        "/api/models",
        "/api/pull",
        "/api/models/{name}",
    } <= set(response.json()["paths"])


def test_list_models() -> None:
    response = TestClient(_server.app).get("/api/models")
    models = {model["name"]: model for model in response.json()}
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
    assert "not found" in response.json()["detail"]
    response = client.post("/api/generate", json={"model": "efficientsam:10m"})
    assert response.status_code == 400

    def fail(request):
        raise RuntimeError("out of GPU memory")

    monkeypatch.setattr(_server.apis, "generate", fail)
    response = client.post("/api/generate", json={"model": "efficientsam:10m"})
    assert response.status_code == 500
    assert response.json()["detail"] == "out of GPU memory"


def test_queued_generates_leave_the_thread_pool_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def blocked(request):
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
