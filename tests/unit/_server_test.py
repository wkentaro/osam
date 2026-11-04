import asyncio
import base64
import json
import pathlib
import threading
import time
import typing
import weakref
from collections.abc import Iterator
from typing import Final

import anyio
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


@pytest.fixture(autouse=True)
def _no_model_loaded(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # The server's state is per process, so a model a test leaves loaded would
    # be in the next test's /api/ps.
    monkeypatch.setattr(_server, "_runtime", _runtime.Runtime())
    yield
    # A keep_alive timer holds its runtime, and the models in it.
    _server._runtime.close()


def _patch_generate(*, monkeypatch: pytest.MonkeyPatch, generate) -> None:
    # Replaces efficientsam:10m's generate.
    model_type = _server._registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(model_type, "generate", generate)


def test_openapi_documents_every_route() -> None:
    response = TestClient(_server.app).get("/openapi.json")
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
    assert response.json()["paths"]["/api/generate"]["post"]["deprecated"]


def test_list_models() -> None:
    response = TestClient(_server.app).get("/api/models")
    models = {model["name"]: model for model in response.json()["models"]}
    assert models["efficientsam:10m"]["size"] > 0
    assert models["efficientsam:10m"]["metadata"]["license_spdx"] == "Apache-2.0"


def test_pull_streams_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    model_type = _server._registry.get_model_type_by_name(name="efficientsam:10m")

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


def test_segment_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_server.app)
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

    def fail(self, request):
        assert self.device == "cpu"
        raise RuntimeError("out of GPU memory")

    monkeypatch.setattr(_server, "_runtime", _runtime.Runtime(device="cpu"))
    _patch_generate(monkeypatch=monkeypatch, generate=fail)
    response = TestClient(_server.app, raise_server_exceptions=False).post(
        "/api/segment", json={"model": "efficientsam:10m", "image": _TINY_IMAGE_B64}
    )
    assert response.status_code == 500
    assert response.json() == {"error": "out of GPU memory"}


def test_queued_segments_leave_the_thread_pool_free(
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

    _patch_generate(monkeypatch=monkeypatch, generate=blocked)
    # Twice, on two event loops, as separate app lifecycles would.
    for _ in range(2):
        release = threading.Event()
        asyncio.run(scenario())


def test_segment_keeps_image_embedding_inside(
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
        "/api/segment", json={"model": "efficientsam:10m", "image": _TINY_IMAGE_B64}
    )
    assert response.status_code == 200, response.text
    assert response.json().keys() == {"annotations", "image_hash"}


def test_segment_reuses_cached_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    _server._runtime._embeddings.clear()
    client = TestClient(_server.app)
    prompt = {"points": [[1280, 800]], "point_labels": [1]}
    body = {"model": "efficientsam:10m", "image": _IMAGE_B64, **prompt}
    first = client.post("/api/segment", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["annotations"][0]["mask"]

    def fail(*args, **kwargs):
        raise AssertionError("encoded the image again")

    model = _server._runtime._loaded["efficientsam:10m"].model
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
    monkeypatch.setattr(
        _server._runtime, "_load", fail
    )  # 412 without loading the model
    missing = client.post("/api/segment", json={**by_hash, "image_hash": "0" * 64})
    assert missing.status_code == 412
    assert "send the image" in missing.json()["error"]
    both = client.post("/api/segment", json={**body, "image_hash": image_hash})
    assert both.status_code == 422
    assert "not both" in both.json()["error"]


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
    assert _requests._parse_duration(value) == seconds


def test_keep_alive_unloads_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(_server.app)
    body = {"model": "efficientsam:10m", "image": _IMAGE_B64}

    response = client.post("/api/segment", json={**body, "keep_alive": -1})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert running["name"] == "efficientsam:10m"
    assert running["device"] in typing.get_args(types.Device)
    assert running["expires_at"] is None

    response = client.post("/api/segment", json={**body, "keep_alive": "0.2s"})
    assert response.status_code == 200, response.text
    (running,) = client.get("/api/ps").json()["models"]
    assert running["expires_at"] > time.time()
    deadline = time.monotonic() + 5
    while client.get("/api/ps").json()["models"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client.get("/api/ps").json() == {"models": []}
    assert not _server._runtime._embeddings

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


def test_two_models_stay_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_server._runtime, "_max_loaded_models", 2)
    client = TestClient(_server.app)

    def segment(model: str) -> None:
        body = {"model": model, "image": _TINY_IMAGE_B64, "keep_alive": -1}
        response = client.post("/api/segment", json=body)
        assert response.status_code == 200, response.text

    def running() -> list[str]:
        return [model["name"] for model in client.get("/api/ps").json()["models"]]

    segment("efficientsam:10m")
    segment("sam2:tiny")
    assert running() == ["sam2:tiny", "efficientsam:10m"]
    sam2 = _server._runtime._loaded["sam2:tiny"].model
    segment("efficientsam:10m")
    assert _server._runtime._loaded["sam2:tiny"].model is sam2  # not reloaded
    assert running() == ["efficientsam:10m", "sam2:tiny"]

    # A full server unloads the least recently used model and its embeddings.
    client.post("/api/load", json={"model": "sam2:tiny", "keep_alive": 0})
    assert [model for model, _ in _server._runtime._embeddings] == ["efficientsam:10m"]
    monkeypatch.setattr(_server._runtime, "_max_loaded_models", 1)
    segment("sam2:tiny")
    assert running() == ["sam2:tiny"]
    assert [model for model, _ in _server._runtime._embeddings] == ["sam2:tiny"]


def test_failed_load_unloads_the_others_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_server._runtime, "_max_loaded_models", 2)
    client = TestClient(_server.app)
    body = {"image": _TINY_IMAGE_B64, "keep_alive": -1}
    response = client.post("/api/segment", json={**body, "model": "efficientsam:10m"})
    assert response.status_code == 200, response.text
    sam2 = _server._registry.get_model_type_by_name(name="sam2:tiny")

    def offline(*args, **kwargs) -> None:
        raise OSError("network is unreachable")

    # A failed download unloads nothing.
    with monkeypatch.context() as patch:
        patch.setattr(sam2, "pull", offline)
        response = TestClient(_server.app, raise_server_exceptions=False).post(
            "/api/segment", json={**body, "model": "sam2:tiny"}
        )
    assert response.json() == {"error": "network is unreachable"}
    assert list(_server._runtime._loaded) == ["efficientsam:10m"]

    loaded_at_each_load: list[list[str]] = []
    failed: list[weakref.ref] = []
    init = sam2.__init__

    def out_of_memory_once(self, *args, **kwargs) -> None:
        loaded_at_each_load.append(list(_server._runtime._loaded))
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
    assert list(_server._runtime._loaded) == ["sam2:tiny"]


def test_cpu_fallback_next_to_a_gpu_model_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_server._runtime, "_max_loaded_models", 2)
    efficientsam = _server._registry.get_model_type_by_name(name="efficientsam:10m")
    monkeypatch.setattr(efficientsam, "device", property(lambda self: "cuda"))
    sam2 = _server._registry.get_model_type_by_name(name="sam2:tiny")
    loads: list[list[str]] = []
    init = sam2.__init__

    def counted(self, *args, **kwargs) -> None:
        loads.append(list(_server._runtime._loaded))
        init(self, *args, **kwargs)

    monkeypatch.setattr(sam2, "__init__", counted)
    _server._runtime._load(model_name="efficientsam:10m")
    _server._runtime._load(model_name="sam2:tiny")  # lands on CPU next to "cuda"
    assert loads == [["efficientsam:10m"], []]
    assert list(_server._runtime._loaded) == ["sam2:tiny"]


def _to_prompt(**fields) -> typing.Any:
    request = _requests.SegmentRequest(model="sam3", image_hash="0", **fields)
    sam3 = _server._registry.get_model_type_by_name(name="sam3")
    return request.to_prompt(model_type=sam3, height=10, width=20)


def test_segment_request_builds_point_labels() -> None:
    prompt = _to_prompt(points=[[1, 2]], point_labels=[0], box=[3, 4, 5, 6])
    assert prompt.points.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert prompt.point_labels.tolist() == [0, 2, 3]
    assert prompt.texts is None

    prompt = _to_prompt(
        texts=["dog"],
        example_boxes=[[0, 0, 1, 1], [2, 2, 3, 3]],
        example_box_labels=[1, 0],
        score_threshold=0.01,
    )
    assert prompt.point_labels.tolist() == [2, 3, 4, 5]
    assert (prompt.texts, prompt.score_threshold) == (["dog"], 0.01)

    # Example boxes alone find every object like them.
    prompt = _to_prompt(example_boxes=[[0, 0, 1, 1]])
    assert (prompt.texts, prompt.point_labels.tolist()) == (["visual"], [2, 3])

    assert _to_prompt() is None  # the model's default prompt
    prompt = _to_prompt(max_annotations=1)  # the default prompt, with it
    assert (prompt.texts, prompt.max_annotations) == (["visual"], 1)


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
def test_segment_request_rejects_invalid_prompts(fields: dict) -> None:
    body = {"model": "sam3", "image_hash": "0", **fields}
    response = TestClient(_server.app).post("/api/segment", json=body)
    assert response.status_code == 422, response.text


def test_generate_keeps_osam_0_10_requests_working(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TestClient(_server.app)
    # As osam 0.10 sent it: a prompt in the models' point labels.
    body = {
        "model": "efficientsam:10m",
        "image": _IMAGE_B64,
        "prompt": {"points": [[1280, 800]], "point_labels": [1]},
    }
    first = client.post("/api/generate", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["annotations"][0]["mask"]
    assert first.json()["image_embedding"] is None

    # The same image again reuses the embedding the server kept, and one sent
    # back is ignored.
    def fail(*args, **kwargs):
        raise AssertionError("encoded the image again")

    model = _server._runtime._loaded["efficientsam:10m"].model
    monkeypatch.setattr(type(model), "encode_image", fail)
    second = client.post(
        "/api/generate", json={**body, "image_embedding": {"embedding": "from 0.10"}}
    )
    assert second.status_code == 200, second.text
    assert second.json()["annotations"] == first.json()["annotations"]

    no_image = client.post("/api/generate", json={**body, "image": None})
    assert no_image.status_code == 400
    assert "send the image" in no_image.json()["error"]


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
