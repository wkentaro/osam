import asyncio
import base64
import contextlib
import json
import logging
import pathlib
import threading
from collections.abc import Iterator
from typing import Final

import anyio
import fastapi
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import osam
from osam import _json
from osam import _requests
from osam import _runtime
from osam import _server
from osam import _types as types

from . import _fakes

_IMAGE_B64: Final = base64.b64encode(
    (
        pathlib.Path(__file__).resolve().parents[2] / "examples/_images/dogs.jpg"
    ).read_bytes()
).decode()
# For requests whose model is replaced, so the image needn't be a real one.
_TINY_IMAGE_B64: Final = _json.image_ndarray_to_b64data(
    ndarray=np.zeros((1, 1, 3), dtype=np.uint8)
)


@pytest.fixture
def runtime() -> Iterator[_runtime.Runtime]:
    runtime = _runtime.Runtime(model_types=[_fakes.model_type(name="fake:latest")])
    yield runtime
    # A keep_alive timer holds its runtime, and the models in it.
    runtime.close()


@pytest.fixture
def app(runtime: _runtime.Runtime) -> fastapi.FastAPI:
    return _server.create_app(runtime=runtime)


def test_openapi_documents_every_route(app: fastapi.FastAPI) -> None:
    response = TestClient(app).get("/openapi.json")
    assert response.status_code == 200, response.text
    assert {
        "/api/segment",
        "/api/load",
        "/api/models",
        "/api/pull",
        "/api/models/{name}",
        "/api/ps",
        "/api/version",
    } <= set(response.json()["paths"])
    assert "/api/generate" not in response.json()["paths"]  # removed in 0.11


def test_list_models(app: fastapi.FastAPI) -> None:
    response = TestClient(app).get("/api/models")
    (model,) = response.json()["models"]
    assert model["name"] == "fake:latest"
    assert model["metadata"]["license_name"] == "MIT"
    assert model["capabilities"] == ["segment"]
    assert model["prompt_kinds"] == ["points"]


def test_pull_streams_progress(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    model_type = runtime.model_type(name="fake")

    def pull(progress, cancel):
        progress("encoder.onnx", 1, 2)
        progress("encoder.onnx", 2, 2)

    monkeypatch.setattr(model_type, "pull", pull)
    client = TestClient(app)
    response = client.post("/api/pull", json={"model": "fake"})
    assert response.headers["content-type"] == "application/x-ndjson"
    assert [json.loads(line) for line in response.text.splitlines()] == [
        {"status": "pulling encoder.onnx", "completed": 1, "total": 2},
        {"status": "pulling encoder.onnx", "completed": 2, "total": 2},
        {"status": "success"},
    ]

    def fail(progress, cancel):
        raise RuntimeError("disk full")

    monkeypatch.setattr(model_type, "pull", fail)
    response = client.post("/api/pull", json={"model": "fake"})
    assert response.text.splitlines()[-1] == json.dumps(
        {"error": "disk full", "code": "pull_failed"}
    )
    response = client.post("/api/pull", json={"model": "missing"})
    assert response.status_code == 404


def test_segment_reports_errors(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    client = TestClient(app)
    response = client.post(
        "/api/segment", json={"model": "missing", "image": _TINY_IMAGE_B64}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["error"]
    assert response.json()["code"] == "model_not_found"
    truncated: str = base64.b64encode(base64.b64decode(_IMAGE_B64)[:1000]).decode()
    for body in [
        {"image": "eA=="},
        {"image": truncated},
    ]:
        response = client.post("/api/segment", json={"model": "fake", **body})
        assert response.status_code == 422, body
        assert "not an image file" in response.json()["error"]
        assert response.json()["code"] == "invalid_request"

    def fail(self, *, image):
        assert self.device == "cpu"
        raise RuntimeError("out of GPU memory")

    monkeypatch.setattr(runtime.model_type(name="fake"), "encode_image", fail)
    with contextlib.closing(
        _runtime.Runtime(device="cpu", model_types=[runtime.model_type(name="fake")])
    ) as cpu:
        response = TestClient(
            _server.create_app(runtime=cpu), raise_server_exceptions=False
        ).post(
            "/api/segment",
            json={"model": "fake", "image": _TINY_IMAGE_B64},
        )
    assert response.status_code == 500
    assert response.json() == {"error": "out of GPU memory", "code": "internal_error"}


def test_queued_segments_leave_the_thread_pool_free(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    def blocked(self, *, image):
        release.wait()
        raise ValueError("released")

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # More than the thread pool's 40 workers.
            body = {"model": "fake", "image": _TINY_IMAGE_B64}
            tasks = [
                asyncio.create_task(client.post("/api/segment", json=body))
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

    monkeypatch.setattr(runtime.model_type(name="fake"), "encode_image", blocked)
    # Twice, on two event loops, as separate app lifecycles would.
    for _ in range(2):
        release = threading.Event()
        asyncio.run(scenario())


def test_segment_by_image_reference(app: fastapi.FastAPI) -> None:
    client = TestClient(app)
    body = {"model": "fake", "image": _TINY_IMAGE_B64}
    response = client.post("/api/segment", json=body)
    assert response.status_code == 200, response.text
    assert response.json().keys() == {"annotations", "image"}

    image_ref = response.json()["image"]
    assert image_ref.startswith("sha256:")
    by_ref = {"model": "fake", "image": image_ref}
    assert client.post("/api/segment", json=by_ref).status_code == 200
    unknown = "sha256:" + "0" * 64
    missing = client.post("/api/segment", json={**by_ref, "image": unknown})
    assert missing.status_code == 412
    assert missing.json()["code"] == "missing_images"
    assert missing.json()["images"] == [unknown]
    bad_ref = client.post("/api/segment", json={**by_ref, "image": "sha256:0"})
    assert bad_ref.status_code == 422
    points = client.post("/api/segment", json={**body, "texts": ["dog"]})
    assert points.status_code == 400
    assert "not texts" in points.json()["error"]


def test_load_and_ps(app: fastapi.FastAPI) -> None:
    client = TestClient(app)
    response = client.post("/api/load", json={"model": "fake", "keep_alive": -1})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert running == {"name": "fake:latest", "device": "cpu", "expires_at": None}
    assert client.post("/api/load", json={"model": "fake", "keep_alive": 0}).is_success
    assert client.get("/api/ps").json() == {"models": []}
    assert client.post("/api/load", json={"model": "missing"}).status_code == 404

    body = {"model": "fake", "image": _TINY_IMAGE_B64}
    for keep_alive in ["1x", "inf", 1e20, None, []]:
        response = client.post("/api/segment", json={**body, "keep_alive": keep_alive})
        assert response.status_code == 422, keep_alive


def test_segment_bounds_the_embedding_cache_in_bytes(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    def encode_image(self, *, image: np.ndarray) -> types.ImageEmbedding:
        return types.ImageEmbedding(
            original_height=1,
            original_width=1,
            embedding=np.zeros(250, dtype=np.float32),  # 1000 bytes
        )

    monkeypatch.setattr(runtime.model_type(name="fake"), "encode_image", encode_image)
    monkeypatch.setattr(runtime, "_embedding_cache_bytes", 2500)
    client = TestClient(app)
    refs: list[str] = []
    for value in range(4):
        image = _json.image_ndarray_to_b64data(
            ndarray=np.full((1, 1, 3), value, dtype=np.uint8)
        )
        response = client.post("/api/segment", json={"model": "fake", "image": image})
        assert response.status_code == 200, response.text
        refs.append(response.json()["image"])
    assert [ref for _, ref in runtime._embeddings] == refs[2:]

    # The newest one stays, even when it alone is over the budget.
    monkeypatch.setattr(runtime, "_embedding_cache_bytes", 1)
    response = client.post("/api/segment", json={"model": "fake", "image": refs[3]})
    assert response.status_code == 200, response.text
    assert [ref for _, ref in runtime._embeddings] == refs[3:]


def test_routing_errors_use_the_error_shape(app: fastapi.FastAPI) -> None:
    client = TestClient(app)
    assert client.get("/api/missing").json() == {
        "error": "Not Found",
        "code": "not_found",
    }
    response = client.post("/api/version")
    assert response.status_code == 405
    assert response.json() == {
        "error": "Method Not Allowed",
        "code": "method_not_allowed",
    }
    assert response.headers["allow"] == "GET"


def test_version(app: fastapi.FastAPI) -> None:
    response = TestClient(app).get("/api/version")
    assert response.json() == {"version": osam.__version__}


@pytest.mark.parametrize(
    ("value", "seconds"),
    [(300, 300), ("10", 10), ("5m", 300), ("1h30m", 5400), ("-1s", -1)],
)
def test_parse_duration(value, seconds) -> None:
    assert _requests._parse_duration(value) == seconds


@pytest.mark.parametrize(
    "fields",
    [
        {"points": [[1, 2]], "texts": ["dog"]},
        {"box": [0, 0, 1, 1], "example_boxes": [[0, 0, 1, 1]]},
        {"points": [[1, 2]]},
        {"points": [[1, 2]], "point_labels": [1, 1]},
        {"example_box_labels": [1]},
        {"point_labels": [2], "points": [[1, 2]]},
        {"box": [0, 0, 1]},
    ],
)
def test_segment_request_rejects_invalid_prompts(
    fields: dict, app: fastapi.FastAPI
) -> None:
    body = {"model": "sam3", "image": "sha256:" + "0" * 64, **fields}
    response = TestClient(app).post("/api/segment", json=body)
    assert response.status_code == 422, response.text


def test_uvicorn_server_logs_to_the_osam_log(
    app: fastapi.FastAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    uvicorn_logger = logging.getLogger("uvicorn")
    monkeypatch.setattr(uvicorn_logger, "handlers", [])
    _server.uvicorn_server(app=app, log_level="warning")
    uvicorn_logger.error("model crashed")
    for handler in uvicorn_logger.handlers:
        handler.close()
    log: pathlib.Path = tmp_path / ".cache/osam/osam.log"
    assert "model crashed" in log.read_text()


def test_a_value_error_is_a_400_on_every_route(
    app: fastapi.FastAPI, runtime: _runtime.Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(**kwargs) -> None:
        raise ValueError("no such thing")

    monkeypatch.setattr(runtime, "load", fail)
    response = TestClient(app).post("/api/load", json={"model": "efficientsam:10m"})
    assert response.status_code == 400
    assert response.json() == {"error": "no such thing", "code": "invalid_request"}
