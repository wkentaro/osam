import asyncio
import threading

import anyio
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from osam import _server
from osam import types


def test_openapi_documents_generate() -> None:
    response = TestClient(_server.app).get("/openapi.json")
    assert response.status_code == 200, response.text
    assert "/api/generate" in response.json()["paths"]
    schemas = response.json()["components"]["schemas"]
    assert "image_embedding" not in schemas["GenerateRequest"]["properties"]
    assert "image_embedding" not in schemas["GenerateResponse"]["properties"]


def test_generate_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_server.app)
    response = client.post("/api/generate", json={"model": "missing"})
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]

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
