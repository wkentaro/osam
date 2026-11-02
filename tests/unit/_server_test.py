import base64
import pathlib
from typing import Final

import pytest
from fastapi.testclient import TestClient

from osam import _server
from osam import apis

_IMAGE_B64: Final = base64.b64encode(
    (
        pathlib.Path(__file__).resolve().parents[2] / "examples/_images/dogs.jpg"
    ).read_bytes()
).decode()


@pytest.fixture
def client() -> TestClient:
    _server._embeddings.clear()
    return TestClient(_server.app)


def test_openapi_documents_every_route(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/generate", "/api/models", "/api/pull", "/api/models/{name}"} <= set(
        paths
    )


def test_list_models(client: TestClient) -> None:
    models = {m["name"]: m for m in client.get("/api/models").json()}
    assert models["efficientsam:10m"]["size"] > 0
    assert models["efficientsam:10m"]["metadata"]["license_spdx"] == "Apache-2.0"


def test_generate_reuses_cached_embedding(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = {"points": [[1280, 800]], "point_labels": [1]}
    body = {"model": "efficientsam:10m", "image": _IMAGE_B64, "prompt": prompt}
    first = client.post("/api/generate", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["image_embedding"] is None
    assert first.json()["annotations"][0]["mask"]

    def fail(*args, **kwargs):
        raise AssertionError("encoded the image again")

    monkeypatch.setattr(type(apis.running_model), "encode_image", fail)
    second = client.post("/api/generate", json={**body, "return_image_embedding": True})
    assert second.status_code == 200, second.text
    assert second.json()["annotations"] == first.json()["annotations"]

    embedding = second.json()["image_embedding"]
    third = client.post(
        "/api/generate",
        json={
            "model": "efficientsam:10m",
            "image_embedding": embedding,
            "prompt": prompt,
        },
    )
    assert third.json()["annotations"] == first.json()["annotations"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"model": "missing", "image": _IMAGE_B64}, 404),
        ({"model": "efficientsam:10m"}, 400),
        ({"model": "efficientsam:10m", "image": _IMAGE_B64, "device": "tpu"}, 422),
    ],
)
def test_generate_rejects_bad_requests(
    client: TestClient, body: dict, status: int
) -> None:
    assert client.post("/api/generate", json=body).status_code == status
