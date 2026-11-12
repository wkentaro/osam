import asyncio
import base64
import contextlib
import json
import logging
import pathlib
import threading
import weakref
from collections.abc import Callable
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
        "/api/track",
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

    def pull(device, progress, cancel):
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

    def fail(device, progress, cancel):
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
    "fields",
    [
        # One the prompt's own validator rejects, which _prompt_test.py covers
        # in full, and one whose type is wrong.
        {"points": [[1, 2]], "texts": ["dog"]},
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


class _FakeTimeline:
    # Object i, from action i, covers frames i and up as a one-pixel mask.

    _get_embedding: Callable[[int], object]
    _num_frames: int
    _applied: list
    _count: int

    def __init__(self, *, get_embedding, num_frames: int, applied: list):
        self._get_embedding = get_embedding
        self._num_frames = num_frames
        self._applied = applied
        self._count = 0

    def apply(self, *, action, on_frame=None) -> None:
        if getattr(action, "object_id", None) == 99:
            raise ValueError("no object 99")
        self._get_embedding(0)
        self._applied.append(action.type)
        self._count += 1
        for frame in range(self._num_frames):
            if on_frame is not None:
                on_frame(frame)

    def annotations(self, *, frame: int) -> list[types.Annotation]:
        return [
            types.annotations_from_mask(mask=np.ones((1, 1), bool), score=1)[
                0
            ].model_copy(update={"object_id": i})
            for i in range(min(self._count, frame + 1))
        ]

    def result(self) -> str:
        return f"sha256:{self._count:064d}"


def _patch_tracking(
    *, monkeypatch: pytest.MonkeyPatch, model_type: type[types.Model]
) -> list[str]:
    # model_type tracks with _FakeTimeline; returns the actions it applies.
    applied: list[str] = []
    monkeypatch.setattr(model_type, "tracks", classmethod(lambda cls: True))
    monkeypatch.setattr(
        model_type,
        "new_timeline",
        lambda self, *, get_embedding, num_frames, height, width: _FakeTimeline(
            get_embedding=get_embedding, num_frames=num_frames, applied=applied
        ),
    )
    return applied


def _track_events(*, client: TestClient, **fields) -> list[dict]:
    response = client.post("/api/track", json={"model": "fake", **fields})
    assert response.status_code == 200, response.text
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["progress"] is True
    # One preparing event, before any position.
    assert [e for e in events if "progress" in e and e.get("frame") is None] == [
        events[0]
    ]
    return events


def _post_track(*, client: TestClient, **fields) -> tuple[list[dict], dict]:
    *events, done = _track_events(client=client, **fields)
    assert done["done"] is True
    frames = [event for event in events if not event.get("progress")]
    return frames, done


def _track_error(*, client: TestClient, **fields) -> dict:
    *progress, error = _track_events(client=client, **fields)
    assert all(event.get("progress") for event in progress)
    assert "error" in error
    return error


def test_track_streams_frames_and_caches_action_prefixes(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    applied = _patch_tracking(
        monkeypatch=monkeypatch, model_type=runtime.model_type(name="fake")
    )
    client = TestClient(app)
    prompt = {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}
    frames, done = _post_track(
        client=client, frames=[_TINY_IMAGE_B64] * 3, actions=[prompt]
    )
    assert [f["frame"] for f in frames] == [0, 1, 2]
    assert [a["object_id"] for a in frames[2]["annotations"]] == [0]
    assert frames[0]["annotations"][0]["mask"]
    assert done == {
        "done": True,
        "objects": [{"object_id": 0, "first_frame": 0, "last_frame": 2}],
        "result": f"sha256:{1:064d}",
        "cache": "miss",
        "images": [_json.image_ref(image=np.zeros((1, 1, 3), np.uint8))] * 3,
    }
    assert applied == ["prompt"]

    # By the references it returned, and the same actions: no model runs.
    ref: str = done["images"][0]
    add = {"type": "add", "frame": 1, "box": [0, 0, 1, 1], "base": done["result"]}
    frames, done = _post_track(
        client=client,
        frames=[ref] * 3,
        actions=[prompt],
        first_frame=1,
        last_frame=1,
        include_masks=False,
    )
    assert done["cache"] == "hit"
    assert [f["frame"] for f in frames] == [1]
    assert frames[0]["annotations"][0]["mask"] is None
    assert applied == ["prompt"]

    # A new action applies to the cached timeline.
    frames, done = _post_track(client=client, frames=[ref] * 3, actions=[prompt, add])
    assert done["cache"] == "partial"
    assert done["objects"][1] == {"object_id": 1, "first_frame": 1, "last_frame": 2}
    assert applied == ["prompt", "add"]

    # Undo returns the earlier result without running the model.
    _, undone = _post_track(client=client, frames=[ref] * 3, actions=[prompt])
    assert undone["cache"] == "hit"
    assert applied == ["prompt", "add"]

    # An edit on another result than the server's, such as after a replay.
    stale = {**add, "base": "sha256:" + "f" * 64}
    error = _track_error(client=client, frames=[ref] * 3, actions=[prompt, stale])
    assert error["status_code"] == 409
    assert error["code"] == "base_changed"
    assert error["result"] == f"sha256:{1:064d}"


def test_track_keeps_copies_of_its_inputs_and_results(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    _patch_tracking(monkeypatch=monkeypatch, model_type=runtime.model_type(name="fake"))
    request = _requests.TrackRequest.model_validate(
        {
            "model": "fake",
            "frames": [np.zeros((1, 1, 3), np.uint8)],
            "actions": [
                {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}
            ],
        }
    )
    (annotation,) = runtime.track(request=request).frames[0].annotations
    assert annotation.mask is not None
    annotation.mask[:] = False
    (annotation,) = runtime.track(request=request).frames[0].annotations
    assert annotation.mask is not None and annotation.mask.all()

    # And keeps a copy of a frame, which the caller may change.
    frame = request.frames[0]
    assert isinstance(frame, np.ndarray)
    ref: str = _json.image_ref(image=frame)
    frame[:] = 255
    assert not runtime._frames[ref].any()


@pytest.mark.parametrize("unload_first", [False, True])
def test_close_releases_video_caches(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime, unload_first: bool
) -> None:
    _patch_tracking(monkeypatch=monkeypatch, model_type=runtime.model_type(name="fake"))
    runtime.track(
        request=_requests.TrackRequest(
            model="fake",
            frames=[np.zeros((2, 3, 3), np.uint8)],
            actions=[types.PromptAction(type="prompt", frame=0, texts=["dog"])],
            keep_alive=-1,
        )
    )
    frame = weakref.ref(next(iter(runtime._frames.values())))
    mask = weakref.ref(next(iter(runtime._tracks.values())).frames[0][0].mask)
    assert frame() is not None and mask() is not None
    if unload_first:
        runtime.load(request=_requests.LoadRequest(model="fake", keep_alive=0))
    runtime.close()
    assert runtime.ps() == []
    assert frame() is None and mask() is None


def test_track_reports_errors(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    client = TestClient(app)
    prompt = {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}
    unknown = "sha256:" + "0" * 64
    body = {"frames": [_TINY_IMAGE_B64, unknown]}

    error = _track_error(client=client, **body, actions=[prompt])
    assert error["status_code"] == 400
    assert "doesn't track" in error["error"]
    error = _track_error(client=client, **body, model="missing", actions=[prompt])
    assert (error["status_code"], error["code"]) == (404, "model_not_found")

    _patch_tracking(monkeypatch=monkeypatch, model_type=runtime.model_type(name="fake"))

    missing = _track_error(client=client, **body, actions=[prompt])
    assert missing["status_code"] == 412
    assert missing["code"] == "missing_images"
    assert missing["images"] == [unknown]

    body["frames"] = [_TINY_IMAGE_B64]
    remove = {"type": "remove", "object_id": 0, "base": "sha256:" + "0" * 64}
    for actions in [[remove], [prompt, prompt], []]:
        response = client.post(
            "/api/track", json={"model": "fake", **body, "actions": actions}
        )
        assert response.status_code == 422, response.text
    error = _track_error(
        client=client,
        **body,
        actions=[prompt, {**remove, "object_id": 99, "base": f"sha256:{1:064d}"}],
    )
    assert error["status_code"] == 400
    assert error["error"] == "no object 99"
    error = _track_error(client=client, **body, actions=[prompt], last_frame=1)
    assert error["status_code"] == 400


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/load", {"model": "fake", "keeep_alive": 0}),
        ("/api/pull", {"model": "fake", "keeep_alive": 0}),
        ("/api/segment", {"model": "fake", "image": _TINY_IMAGE_B64, "text": "person"}),
        (
            "/api/track",
            {
                "model": "fake",
                "frames": [_TINY_IMAGE_B64],
                "actions": [{"type": "prompt", "frame": 0, "texts": ["person"]}],
                "text": "person",
            },
        ),
        (
            "/api/track",
            {
                "model": "fake",
                "frames": [_TINY_IMAGE_B64],
                "actions": [{"type": "prompt", "frame": 0, "text": "person"}],
            },
        ),
    ],
)
def test_requests_reject_unknown_fields(
    app: fastapi.FastAPI, runtime: _runtime.Runtime, path: str, body: dict
) -> None:
    response = TestClient(app).post(path, json=body)
    assert response.status_code == 422, response.text
    assert response.json()["code"] == "invalid_request"
    assert "Extra inputs" in response.json()["error"]
    assert runtime.ps() == []


def test_sam3_tracks() -> None:
    with contextlib.closing(_runtime.Runtime()) as runtime:
        app = _server.create_app(runtime=runtime)
        models = TestClient(app).get("/api/models").json()["models"]
    (sam3,) = (m for m in models if m["name"] == "sam3:latest")
    assert sam3["capabilities"] == ["segment", "track"]


def test_track_bounds_its_results_with_long_histories(
    monkeypatch: pytest.MonkeyPatch, app: fastapi.FastAPI, runtime: _runtime.Runtime
) -> None:
    _patch_tracking(monkeypatch=monkeypatch, model_type=runtime.model_type(name="fake"))
    monkeypatch.setattr(runtime, "_track_cache_bytes", 16384)
    client = TestClient(app)
    actions: list[dict] = [
        {"type": "prompt", "frame": 0, "points": [[0, 0]], "point_labels": [1]}
    ]
    for _ in range(40):
        _, done = _post_track(
            client=client, frames=[_TINY_IMAGE_B64] * 3, actions=actions
        )
        actions.append(
            {
                "type": "refine",
                "frame": 0,
                "object_id": 0,
                "points": [[0, 0]],
                "point_labels": [0],
                "base": done["result"],
            }
        )
    # Each result costs its frames, not its actions, which the key holds by
    # digest.
    assert 1 < len(runtime._tracks) < 40
    assert sum(track.nbytes for track in runtime._tracks.values()) <= 16384
    # An annotation costs more than its mask's few pixels.
    assert all(
        track.nbytes > 1024 * sum(len(f) for f in track.frames) > 0
        for track in runtime._tracks.values()
    )
    assert all(len(actions) == 64 for _, actions in runtime._tracks)


def test_track_openapi_describes_ndjson_and_structured_http_errors(
    app: fastapi.FastAPI,
) -> None:
    document = TestClient(app).get("/openapi.json").json()
    responses = document["paths"]["/api/track"]["post"]["responses"]
    assert set(responses["200"]["content"]) == {"application/x-ndjson"}
    schema = responses["200"]["content"]["application/x-ndjson"]["schema"]
    assert {item["$ref"].rsplit("/", 1)[-1] for item in schema["anyOf"]} == {
        "TrackProgress",
        "TrackFrame",
        "TrackDone",
        "TrackError",
    }
    assert set(responses) == {"200", "422"}
    assert set(responses["422"]["content"]) == {"application/json"}
    error = responses["422"]["content"]["application/json"]["schema"]
    assert set(error["required"]) == {"error", "code"}
    assert (
        "progress"
        not in document["components"]["schemas"]["TrackRequest"]["properties"]
    )


def test_track_callback_failure_discards_partial_timeline_and_releases_callback(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    applied = _patch_tracking(
        monkeypatch=monkeypatch, model_type=runtime.model_type(name="fake")
    )
    request = _requests.TrackRequest(
        model="fake",
        frames=[np.zeros((1, 1, 3), np.uint8)] * 3,
        actions=[
            types.PromptAction(
                type="prompt", frame=0, points=[(0, 0)], point_labels=[1]
            )
        ],
    )

    class Stop(BaseException):
        pass

    class Progress:
        def __call__(self, event: types.TrackProgress) -> None:
            if event.frame == 1:
                raise Stop()

    callback = Progress()
    ref = weakref.ref(callback)
    with pytest.raises(Stop):
        runtime.track(request=request, progress=callback)
    assert runtime._timeline is None
    assert runtime._track_images == []
    assert not runtime._tracks
    del callback
    assert ref() is None
    assert runtime.track(request=request).done.cache == "miss"
    assert applied == ["prompt", "prompt"]

    class Success:
        def __call__(self, event: types.TrackProgress) -> None:
            pass

    runtime._tracks.clear()
    runtime._timeline = None
    success = Success()
    ref = weakref.ref(success)
    runtime.track(request=request, progress=success)
    del success
    assert ref() is None


def test_track_stream_logs_unexpected_failure(
    app: fastapi.FastAPI,
    runtime: _runtime.Runtime,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail(**kwargs):
        raise RuntimeError("model failed")

    monkeypatch.setattr(runtime, "track", fail)
    logger = logging.getLogger("uvicorn.error")
    logger.addHandler(caplog.handler)
    try:
        response = TestClient(app).post(
            "/api/track",
            json={
                "model": "fake",
                "frames": [_TINY_IMAGE_B64],
                "actions": [
                    {
                        "type": "prompt",
                        "frame": 0,
                        "points": [[0, 0]],
                        "point_labels": [1],
                    }
                ],
            },
        )
    finally:
        logger.removeHandler(caplog.handler)
    *_, error = [json.loads(line) for line in response.text.splitlines()]
    assert error == {
        "error": "model failed",
        "code": "internal_error",
        "status_code": 500,
    }
    assert "Video tracking failed" in caplog.text
    assert "RuntimeError: model failed" in caplog.text
