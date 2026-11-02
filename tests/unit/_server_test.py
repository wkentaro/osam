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
        lambda request: types.GenerateResponse(
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
