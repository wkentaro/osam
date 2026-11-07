import asyncio
import base64
import contextlib
import json
import logging
import pathlib
import threading
import time
import typing
import weakref
from collections.abc import Iterator
from typing import Final

import anyio
import fastapi
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from osam import _json
from osam import _requests
from osam import _runtime
from osam import _server
from osam import _types as types

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
    runtime = _runtime.Runtime()
    yield runtime
    # A keep_alive timer holds its runtime, and the models in it.
    runtime.close()


@pytest.fixture
def app(runtime: _runtime.Runtime) -> fastapi.FastAPI:
    return _server.create_app(runtime=runtime)


def _patch_encode_image(*, monkeypatch: pytest.MonkeyPatch, encode_image) -> None:
    # Replaces efficientsam:10m's encode_image.
    model_type = _server._registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "encode_image", encode_image)


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
    models = {model["name"]: model for model in response.json()["models"]}
    assert models["efficientsam:10m"]["size"] > 0
    assert models["efficientsam:10m"]["metadata"]["license_spdx"] == "Apache-2.0"


def test_pull_streams_progress(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI
) -> None:
    model_type = _server._registry.get_model_type_by_name(name="efficientsam:10m")

    def pull(progress, cancel):
        progress("encoder.onnx", 1, 2)
        progress("encoder.onnx", 2, 2)

    monkeypatch.setattr(model_type, "pull", pull)
    client = TestClient(app)
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


def test_segment_reports_errors(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI
) -> None:
    client = TestClient(app)
    response = client.post(
        "/api/segment", json={"model": "missing", "image": _TINY_IMAGE_B64}
    )
    assert response.status_code == 404
    assert "not found" in response.json()["error"]
    truncated: str = base64.b64encode(base64.b64decode(_IMAGE_B64)[:1000]).decode()
    for body in [
        {"image": "eA=="},
        {"image": truncated},
    ]:
        response = client.post(
            "/api/segment", json={"model": "efficientsam:10m", **body}
        )
        assert response.status_code == 422, body
        assert "not an image file" in response.json()["error"]

    def fail(self, *, image):
        assert self.device == "cpu"
        raise RuntimeError("out of GPU memory")

    _patch_encode_image(monkeypatch=monkeypatch, encode_image=fail)
    with contextlib.closing(_runtime.Runtime(device="cpu")) as cpu:
        response = TestClient(
            _server.create_app(runtime=cpu), raise_server_exceptions=False
        ).post(
            "/api/segment",
            json={"model": "efficientsam:10m", "image": _TINY_IMAGE_B64},
        )
    assert response.status_code == 500
    assert response.json() == {"error": "out of GPU memory"}


def test_queued_segments_leave_the_thread_pool_free(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI
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
            body = {"model": "efficientsam:10m", "image": _TINY_IMAGE_B64}
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

    _patch_encode_image(monkeypatch=monkeypatch, encode_image=blocked)
    # Twice, on two event loops, as separate app lifecycles would.
    for _ in range(2):
        release = threading.Event()
        asyncio.run(scenario())


def test_segment_keeps_image_embedding_inside(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI
) -> None:
    embedding = types.ImageEmbedding(
        original_height=1,
        original_width=1,
        embedding=np.zeros((1, 1, 1), dtype=np.float32),
    )
    _patch_encode_image(
        monkeypatch=monkeypatch, encode_image=lambda self, *, image: embedding
    )
    model_type = _server._registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "decode", lambda self, *, embedding, prompt: [])
    client = TestClient(app)

    response = client.post(
        "/api/segment", json={"model": "efficientsam:10m", "image": _TINY_IMAGE_B64}
    )
    assert response.status_code == 200, response.text
    assert response.json().keys() == {"annotations", "image_hash"}


def test_segment_reuses_cached_embedding(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    client = TestClient(app)
    prompt = {"points": [[1280, 800]], "point_labels": [1]}
    body = {"model": "efficientsam:10m", "image": _IMAGE_B64, **prompt}
    first = client.post("/api/segment", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["annotations"][0]["mask"]

    def fail(*args, **kwargs):
        raise AssertionError("encoded the image again")

    model = runtime._loaded["efficientsam:10m"].model
    monkeypatch.setattr(type(model), "encode_image", fail)
    second = client.post("/api/segment", json=body)
    assert second.status_code == 200, second.text
    assert second.json()["annotations"] == first.json()["annotations"]

    # By the hash the server returned, in place of the image.
    image_hash = first.json()["image_hash"]
    assert second.json()["image_hash"] == image_hash
    by_hash = {**body, "image": None, "image_hash": image_hash}
    third = client.post("/api/segment", json=by_hash)
    assert third.status_code == 200, third.text
    assert third.json()["annotations"] == first.json()["annotations"]
    assert third.json()["image_hash"] == image_hash
    monkeypatch.setattr(runtime, "_load", fail)  # 412 without loading the model
    missing = client.post("/api/segment", json={**by_hash, "image_hash": "0" * 64})
    assert missing.status_code == 412
    assert "send the image" in missing.json()["error"]
    both = client.post("/api/segment", json={**body, "image_hash": image_hash})
    assert both.status_code == 422
    assert "not both" in both.json()["error"]


def test_routing_errors_use_the_error_shape(app: fastapi.FastAPI) -> None:
    client = TestClient(app)
    assert client.get("/api/missing").json() == {"error": "Not Found"}
    response = client.post("/api/version")
    assert response.status_code == 405
    assert response.json() == {"error": "Method Not Allowed"}
    assert response.headers["allow"] == "GET"


def test_version(app: fastapi.FastAPI) -> None:
    response = TestClient(app).get("/api/version")
    assert response.json() == {"version": _server.__version__}


@pytest.mark.parametrize(
    ("value", "seconds"),
    [(300, 300), ("10", 10), ("5m", 300), ("1h30m", 5400), ("-1s", -1)],
)
def test_parse_duration(value, seconds) -> None:
    assert _requests._parse_duration(value) == seconds


def test_keep_alive_unloads_the_model(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    client = TestClient(app)
    body = {"model": "efficientsam:10m", "image": _IMAGE_B64}

    response = client.post("/api/segment", json={**body, "keep_alive": -1})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert running["name"] == "efficientsam:10m"
    assert running["device"] in typing.get_args(types.Device)
    assert running["expires_at"] is None

    # The test fires the unload timer itself, so a slow runner can't unload
    # the model before /api/ps sees it.
    timers: list[threading.Timer] = []

    class _Timer(threading.Timer):
        def start(self) -> None:
            timers.append(self)

    monkeypatch.setattr(_runtime.threading, "Timer", _Timer)
    requested_at = time.time()
    response = client.post("/api/segment", json={**body, "keep_alive": "0.2s"})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert requested_at + 0.2 <= running["expires_at"] <= time.time() + 0.2
    (timer,) = timers
    assert timer.interval == 0.2
    timer.function()
    assert client.get("/api/ps").json() == {"models": []}
    assert not runtime._embeddings

    # /api/load loads a model and sets its keep_alive; 0 unloads it, and
    # leaves another loaded model alone.
    response = client.post("/api/load", json={"model": "efficientsam:10m"})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert running["expires_at"] > time.time()
    unload = {"model": "efficientsam:10m", "keep_alive": 0}
    assert client.post("/api/load", json={**unload, "model": "sam2:tiny"}).is_success
    assert client.get("/api/ps").json()["models"]
    assert client.post("/api/load", json=unload).is_success
    assert client.get("/api/ps").json() == {"models": []}
    assert client.post("/api/load", json={"model": "missing"}).status_code == 404

    for keep_alive in ["1x", "inf", 1e20, None, []]:
        response = client.post("/api/segment", json={**body, "keep_alive": keep_alive})
        assert response.status_code == 422, keep_alive


def test_two_models_stay_loaded(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    monkeypatch.setattr(runtime, "_max_loaded_models", 2)
    client = TestClient(app)

    def segment(model: str) -> None:
        body = {"model": model, "image": _TINY_IMAGE_B64, "keep_alive": -1}
        response = client.post("/api/segment", json=body)
        assert response.status_code == 200, response.text

    def running() -> list[str]:
        return [model["name"] for model in client.get("/api/ps").json()["models"]]

    segment("efficientsam:10m")
    segment("sam2:tiny")
    assert running() == ["sam2:tiny", "efficientsam:10m"]
    sam2 = runtime._loaded["sam2:tiny"].model
    segment("efficientsam:10m")
    assert runtime._loaded["sam2:tiny"].model is sam2  # not reloaded
    assert running() == ["efficientsam:10m", "sam2:tiny"]

    # A full server unloads the least recently used model and its embeddings.
    client.post("/api/load", json={"model": "sam2:tiny", "keep_alive": 0})
    assert [model for model, _ in runtime._embeddings] == ["efficientsam:10m"]
    monkeypatch.setattr(runtime, "_max_loaded_models", 1)
    segment("sam2:tiny")
    assert running() == ["sam2:tiny"]
    assert [model for model, _ in runtime._embeddings] == ["sam2:tiny"]


def test_failed_load_unloads_the_others_and_retries(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    monkeypatch.setattr(runtime, "_max_loaded_models", 2)
    client = TestClient(app)
    body = {"image": _TINY_IMAGE_B64, "keep_alive": -1}
    response = client.post("/api/segment", json={**body, "model": "efficientsam:10m"})
    assert response.status_code == 200, response.text
    sam2 = _server._registry.get_model_type_by_name(name="sam2:tiny")

    def offline(*args, **kwargs) -> None:
        raise OSError("network is unreachable")

    # A failed download unloads nothing.
    with monkeypatch.context() as patch:
        patch.setattr(sam2, "pull", offline)
        response = TestClient(app, raise_server_exceptions=False).post(
            "/api/segment", json={**body, "model": "sam2:tiny"}
        )
    assert response.json() == {"error": "network is unreachable"}
    assert list(runtime._loaded) == ["efficientsam:10m"]

    loaded_at_each_load: list[list[str]] = []
    failed: list[weakref.ref] = []
    init = sam2.__init__

    def out_of_memory_once(self, *args, **kwargs) -> None:
        loaded_at_each_load.append(list(runtime._loaded))
        if not failed:
            failed.append(weakref.ref(self))
            raise RuntimeError("out of memory")
        # The failed load, and the memory it holds, is gone before the retry.
        assert failed[0]() is None
        init(self, *args, **kwargs)

    monkeypatch.setattr(sam2, "__init__", out_of_memory_once)
    response = client.post("/api/segment", json={**body, "model": "sam2:tiny"})
    assert response.status_code == 200, response.text
    assert loaded_at_each_load == [["efficientsam:10m"], []]
    assert list(runtime._loaded) == ["sam2:tiny"]


def test_cpu_fallback_next_to_a_gpu_model_retries(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    monkeypatch.setattr(runtime, "_max_loaded_models", 2)
    efficientsam = _server._registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(efficientsam, "device", property(lambda self: "cuda"))
    sam2 = _server._registry.get_model_type_by_name(name="sam2:tiny")
    loads: list[list[str]] = []
    init = sam2.__init__

    def counted(self, *args, **kwargs) -> None:
        loads.append(list(runtime._loaded))
        init(self, *args, **kwargs)

    monkeypatch.setattr(sam2, "__init__", counted)
    runtime._load(model_name="efficientsam:10m")
    runtime._load(model_name="sam2:tiny")  # lands on CPU next to "cuda"
    assert loads == [["efficientsam:10m"], []]
    assert list(runtime._loaded) == ["sam2:tiny"]


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
    body = {"model": "sam3", "image_hash": "0", **fields}
    response = TestClient(app).post("/api/segment", json=body)
    assert response.status_code == 422, response.text


def test_keep_alive_frees_the_model_before_another_load() -> None:
    runtime = _runtime.Runtime()
    locked_when_freed: list[bool] = []

    class Model:
        def __del__(self) -> None:
            locked_when_freed.append(runtime._lock.locked())

    runtime._loaded["m"] = _runtime._Loaded(model=typing.cast(typing.Any, Model()))
    runtime._keep_alive(model_name="m", seconds=0.01)
    deadline = time.monotonic() + 5
    while not locked_when_freed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert locked_when_freed == [True]


def test_uvicorn_server_logs_to_the_osam_log(
    app: fastapi.FastAPI, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    uvicorn_logger = logging.getLogger("uvicorn")
    monkeypatch.setattr(uvicorn_logger, "handlers", [])
    _server.uvicorn_server(app=app, log_level="warning")
    uvicorn_logger.error("model crashed")
    for handler in uvicorn_logger.handlers:
        handler.close()
    log: pathlib.Path = tmp_path / ".cache/osam/osam.log"
    assert "model crashed" in log.read_text()
