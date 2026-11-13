import contextlib
import threading
import time
import weakref
from collections.abc import Iterator
from typing import Final

import numpy as np
import onnxruntime
import pytest

from osam import _errors
from osam import _requests
from osam import _runtime
from osam import types

from . import _fakes

_IMAGE: Final = np.zeros((1, 1, 3), dtype=np.uint8)


@pytest.fixture
def runtime() -> Iterator[_runtime.Runtime]:
    runtime = _runtime.Runtime(
        max_loaded_models=2,
        model_types=[_fakes.model_type(name=name) for name in ("a:latest", "b:1")],
    )
    yield runtime
    # A keep_alive timer holds its runtime, and the models in it.
    runtime.close()


def _segment(*, runtime: _runtime.Runtime, **fields) -> _requests.SegmentResponse:
    return runtime.segment(
        request=_requests.SegmentRequest.model_validate(
            {"image": _IMAGE, "keep_alive": -1} | fields
        )
    )


def _running(*, runtime: _runtime.Runtime) -> list[str]:
    return [model.name for model in runtime.ps()]


def test_unknown_model_raises(runtime: _runtime.Runtime) -> None:
    assert [info.name for info in runtime.list()] == ["a:latest", "b:1"]
    with pytest.raises(_errors.UnknownModelError):
        _segment(runtime=runtime, model="efficientsam:10m")  # registered, but not here


def test_keep_alive_unloads_the_model(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    _segment(runtime=runtime, model="a")
    (running,) = runtime.ps()
    assert (running.name, running.device, running.expires_at) == (
        "a:latest",
        "cpu",
        None,
    )

    # The test fires the unload timer itself, so a slow runner can't unload
    # the model before ps sees it.
    timers: list[threading.Timer] = []

    class _Timer(threading.Timer):
        def start(self) -> None:
            timers.append(self)

    monkeypatch.setattr(_runtime.threading, "Timer", _Timer)
    requested_at = time.time()
    response = _segment(runtime=runtime, model="a", keep_alive="0.2s")
    (running,) = runtime.ps()
    assert running.expires_at is not None
    assert requested_at + 0.2 <= running.expires_at <= time.time() + 0.2
    (timer,) = timers
    assert timer.interval == 0.2
    timer.function()
    assert runtime.ps() == []
    # Its embeddings went with it.
    with pytest.raises(_errors.UnknownImageError):
        _segment(runtime=runtime, model="a", image=response.image)

    # load keeps a model loaded; keep_alive 0 unloads it, and only it.
    runtime.load(request=_requests.LoadRequest(model="a"))
    expires_at = runtime.ps()[0].expires_at
    assert expires_at is not None and expires_at > time.time()
    runtime.load(request=_requests.LoadRequest(model="b:1", keep_alive=0))
    assert _running(runtime=runtime) == ["a:latest"]
    runtime.load(request=_requests.LoadRequest(model="a", keep_alive=0))
    assert runtime.ps() == []


def test_segment_reuses_cached_embedding(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    first = _segment(runtime=runtime, model="a")
    model_type = runtime.model_type(name="a")

    def fail(*args, **kwargs):
        raise AssertionError("encoded the image again")

    monkeypatch.setattr(model_type, "encode_image", fail)
    assert _segment(runtime=runtime, model="a").image == first.image
    assert _segment(runtime=runtime, model="a", image=first.image) == first

    # An unknown reference raises before loading, so a retry with the image loads once.
    runtime.load(request=_requests.LoadRequest(model="a", keep_alive=0))
    monkeypatch.setattr(model_type, "__init__", fail)
    with pytest.raises(_errors.UnknownImageError, match="send the images"):
        _segment(runtime=runtime, model="a", image="sha256:" + "0" * 64)


def test_two_models_stay_loaded(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    loaded: list[str] = []
    for name in ["a", "b:1"]:
        monkeypatch.setattr(
            runtime.model_type(name=name),
            "__init__",
            lambda self, device=None, auto_pull=True: loaded.append(self.name),
        )

    _segment(runtime=runtime, model="a")
    _segment(runtime=runtime, model="b:1")
    _segment(runtime=runtime, model="a")
    assert loaded == ["a:latest", "b:1"]  # not reloaded
    assert _running(runtime=runtime) == ["a:latest", "b:1"]

    # When full, the least recently used model unloads first.
    with contextlib.closing(
        _runtime.Runtime(
            model_types=[runtime.model_type(name="a"), runtime.model_type(name="b:1")]
        )
    ) as full:
        _segment(runtime=full, model="a")
        _segment(runtime=full, model="b:1")
        assert _running(runtime=full) == ["b:1"]


def test_load_does_not_pull_a_model_already_on_disk(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    b = runtime.model_type(name="b:1")

    def pulled(*args, **kwargs) -> None:
        raise AssertionError("pulled")

    monkeypatch.setattr(b, "pull", pulled)
    _segment(runtime=runtime, model="b:1")
    assert _running(runtime=runtime) == ["b:1"]


def test_load_without_auto_pull_raises_before_pulling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _fakes.model_type(name="a:latest")
    missing = types.Blob(url="https://example.com/a.onnx", hash="sha256:" + "0" * 64)
    monkeypatch.setattr(a, "_blobs", {"a": missing})
    monkeypatch.setattr(a, "pull", lambda *args, **kwargs: pytest.fail("pulled"))
    with contextlib.closing(
        _runtime.Runtime(device="cpu", auto_pull=False, model_types=[a])
    ) as runtime:
        with pytest.raises(_errors.NotPulledError) as e:
            _segment(runtime=runtime, model="a")
    assert e.value.device == "cpu"


def test_failed_load_unloads_the_others_and_retries(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    _segment(runtime=runtime, model="a")
    b = runtime.model_type(name="b:1")

    def offline(*args, **kwargs) -> None:
        raise OSError("network is unreachable")

    # A failed download unloads nothing.
    with monkeypatch.context() as patch:
        patch.setattr(b, "get_size", classmethod(lambda cls, device=None: None))
        patch.setattr(b, "pull", offline)
        with pytest.raises(OSError, match="unreachable"):
            _segment(runtime=runtime, model="b:1")
    assert _running(runtime=runtime) == ["a:latest"]

    running_at_each_load: list[list[str]] = []
    failed: list[weakref.ref] = []
    init = b.__init__

    def out_of_memory_once(self, *args, **kwargs) -> None:
        running_at_each_load.append(_running(runtime=runtime))
        if not failed:
            failed.append(weakref.ref(self))
            raise RuntimeError("out of memory")
        # The failed load, and the memory it holds, is gone before the retry.
        assert failed[0]() is None
        init(self, *args, **kwargs)

    monkeypatch.setattr(b, "__init__", out_of_memory_once)
    _segment(runtime=runtime, model="b:1")
    assert running_at_each_load == [["a:latest"], []]
    assert _running(runtime=runtime) == ["b:1"]


def test_cpu_fallback_next_to_a_gpu_model_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gpu = _fakes.model_type(name="gpu:latest", device="cuda")
    cpu = _fakes.model_type(name="cpu:latest")
    with contextlib.closing(
        _runtime.Runtime(max_loaded_models=2, model_types=[gpu, cpu])
    ) as runtime:
        running_at_each_load: list[list[str]] = []
        monkeypatch.setattr(
            cpu,
            "__init__",
            lambda self, device=None, auto_pull=True: running_at_each_load.append(
                _running(runtime=runtime)
            ),
        )
        _segment(runtime=runtime, model="gpu")
        _segment(runtime=runtime, model="cpu")  # lands on CPU next to "cuda"
        # Loaded again once the GPU model is gone, so it can land on the GPU.
        assert running_at_each_load == [["gpu:latest"], []]
        assert _running(runtime=runtime) == ["cpu:latest"]


def test_keep_alive_frees_the_model_before_another_load(
    monkeypatch: pytest.MonkeyPatch, runtime: _runtime.Runtime
) -> None:
    locked_when_freed: list[bool] = []
    monkeypatch.setattr(
        runtime.model_type(name="a"),
        "__del__",
        lambda self: locked_when_freed.append(runtime._lock.locked()),
        raising=False,
    )
    _segment(runtime=runtime, model="a", keep_alive="0.01s")
    deadline = time.monotonic() + 5
    while not locked_when_freed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert locked_when_freed == [True]


def test_unavailable_device_raises_when_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        _runtime.Runtime(device="cuda", model_types=[])
    with pytest.raises(ValueError, match="unknown device"):
        _runtime.Runtime(device="gpu", model_types=[])  # type: ignore[arg-type]
