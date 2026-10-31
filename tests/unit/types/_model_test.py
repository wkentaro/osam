import os
import pathlib
import sys
import types
from typing import Any
from typing import Iterator

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import helper
from onnx import numpy_helper

from osam.types import _model
from osam.types._blob import Blob
from osam.types._model import resolve_providers


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    calls: list[str] = []
    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(ort, "preload_dlls", lambda: calls.append("preload"))
    monkeypatch.setattr(
        ort,
        "create_and_register_allocator_v2",
        lambda *args: calls.append("register"),
    )
    _model._setup_cuda.cache_clear()
    yield calls
    _model._setup_cuda.cache_clear()


def test_resolve_providers_unset_ends_with_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    assert resolve_providers()[-1] == "CPUExecutionProvider"


def test_resolve_providers_env_keeps_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "OSAM_ONNX_PROVIDERS", " CoreMLExecutionProvider , CPUExecutionProvider "
    )
    assert resolve_providers() == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_blank_env_is_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_ONNX_PROVIDERS", " , ")
    assert resolve_providers()[-1] == "CPUExecutionProvider"


def test_setup_cuda_skips_build_without_cuda(
    monkeypatch: pytest.MonkeyPatch, fake_cuda: list[str]
) -> None:
    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CPUExecutionProvider"]
    )
    assert _model._setup_cuda() is False
    assert fake_cuda == []


def test_setup_cuda_registers_shared_arena_once(fake_cuda: list[str]) -> None:
    assert _model._setup_cuda() is True
    assert _model._setup_cuda() is True
    assert fake_cuda == ["preload", "register"]


def test_setup_cuda_keeps_preload_output_off_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_cuda: list[str],
) -> None:
    ort: Any = onnxruntime
    monkeypatch.setattr(ort, "preload_dlls", lambda: print("Failed to load libcudnn"))
    _model._setup_cuda()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Failed to load libcudnn" in captured.err


def test_load_session_keeps_providers_when_cuda_setup_fails(
    monkeypatch: pytest.MonkeyPatch, fake_cuda: list[str]
) -> None:
    def fail(*args: Any) -> None:
        raise RuntimeError("CUDA Provider not available")

    sessions: list[tuple[Any, Any]] = []
    ort: Any = onnxruntime
    monkeypatch.setattr(ort, "create_and_register_allocator_v2", fail)
    monkeypatch.setattr(
        ort,
        "InferenceSession",
        lambda path, sess_options=None, providers=None: sessions.append(
            (sess_options, providers)
        ),
    )
    providers = [
        "AzureExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    blob: Any = type("FakeBlob", (), {"path": "model.onnx"})()

    _model._load_inference_session(blob=blob, providers=providers)

    [(sess_options, used_providers)] = sessions
    assert used_providers == providers
    with pytest.raises(RuntimeError, match="does not have configuration"):
        sess_options.get_session_config_entry("session.use_env_allocators")


@pytest.fixture
def fake_sessions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    sessions: list[tuple[str, Any]] = []

    class FakeSession:
        def __init__(self, path: str, sess_options: Any, providers: Any = None) -> None:
            if providers is None:
                providers = [d.ep_name for d in sess_options.devices]
                providers.append("CPUExecutionProvider")
            sessions.append((path, providers))
            self._providers = [p if isinstance(p, str) else p[0] for p in providers]

        def get_providers(self) -> list[str]:
            return self._providers

        def disable_fallback(self) -> None:
            sessions.append((self._providers[0], "fallback disabled"))

    ort: Any = onnxruntime
    monkeypatch.setattr(ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(_model.Model, "pull", classmethod(lambda cls: None))
    return sessions


def test_device_coreml_runs_only_coreml_blobs_on_coreml(
    monkeypatch: pytest.MonkeyPatch, fake_sessions: list[tuple[str, Any]]
) -> None:
    from osam._models.sam3._models import Sam3

    monkeypatch.setattr(_model, "_coreml_model_path", lambda blob: f"{blob.path}.t")
    Sam3(device="coreml")

    coreml = [
        (path, providers[0][1]["ModelCacheDirectory"])
        for path, providers in fake_sessions
        if providers[0][0] == "CoreMLExecutionProvider"
    ]
    assert coreml == [
        (
            f"{Sam3._blobs[key].path}.t",
            os.path.join(Sam3._blobs[key].coreml_cache_path, onnxruntime.__version__),
        )
        for key in ("image_encoder", "language_encoder", "decoder")
    ]
    disabled = [p for p, providers in fake_sessions if providers == "fallback disabled"]
    assert disabled == [
        "CoreMLExecutionProvider",
        "CoreMLExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_device_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch, fake_sessions: list[tuple[str, Any]]
) -> None:
    from osam._models.sam3._models import Sam3

    def pull(cls: Any) -> None:
        raise AssertionError("downloaded before checking the device")

    monkeypatch.setattr(_model.Model, "pull", classmethod(pull))

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is unavailable"):
        Sam3(device="cuda")
    with pytest.raises(ValueError, match="device must be one of"):
        Sam3(device="gpu")  # type: ignore[invalid-argument-type]


def test_device_dropped_provider_raises(
    monkeypatch: pytest.MonkeyPatch, fake_sessions: list[tuple[str, Any]]
) -> None:
    from osam._models.sam3._models import Sam3

    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort.InferenceSession, "get_providers", lambda self: ["CPUExecutionProvider"]
    )
    monkeypatch.setattr(_model, "_coreml_model_path", lambda blob: blob.path)
    with pytest.raises(RuntimeError, match="CoreMLExecutionProvider failed to load"):
        Sam3(device="coreml")


@pytest.fixture
def fake_webgpu(
    monkeypatch: pytest.MonkeyPatch, fake_sessions: list[tuple[str, Any]]
) -> Iterator[list[Any]]:
    devices: list[Any] = [types.SimpleNamespace(ep_name="CPUExecutionProvider")]
    plugin = types.SimpleNamespace(
        get_library_path=lambda: "libwebgpu.so",
        get_ep_name=lambda: "WebGpuExecutionProvider",
    )

    class FakeSessionOptions:
        def add_provider_for_devices(self, devices: Any, options: Any) -> None:
            self.devices = devices

    ort: Any = onnxruntime
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", plugin)
    monkeypatch.setattr(ort, "register_execution_provider_library", lambda *a: None)
    monkeypatch.setattr(ort, "get_ep_devices", lambda: devices)
    monkeypatch.setattr(ort, "SessionOptions", FakeSessionOptions)
    _model._webgpu_device.cache_clear()
    yield devices
    _model._webgpu_device.cache_clear()


def test_device_webgpu_runs_every_blob_on_webgpu(
    fake_webgpu: list[Any], fake_sessions: list[tuple[str, Any]]
) -> None:
    from osam._models.sam3._models import Sam3

    fake_webgpu.append(types.SimpleNamespace(ep_name="WebGpuExecutionProvider"))
    Sam3(device="webgpu")

    disabled = [p for p, providers in fake_sessions if providers == "fallback disabled"]
    assert disabled == ["WebGpuExecutionProvider"] * len(Sam3._blobs)


def test_device_webgpu_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    from osam._models.sam3._models import Sam3

    with pytest.raises(RuntimeError, match="WebGpuExecutionProvider found no GPU"):
        Sam3(device="webgpu")

    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", None)
    _model._webgpu_device.cache_clear()
    with pytest.raises(RuntimeError, match=r"pip install 'osam\[webgpu\]'"):
        Sam3(device="webgpu")


def test_resolve_providers_picks_webgpu_after_cuda(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    assert resolve_providers() == ["CPUExecutionProvider"]  # no GPU listed

    fake_webgpu.append(types.SimpleNamespace(ep_name="WebGpuExecutionProvider"))
    _model._webgpu_device.cache_clear()
    assert resolve_providers() == ["WebGpuExecutionProvider", "CPUExecutionProvider"]

    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert resolve_providers() == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_resolve_providers_without_webgpu_plugin_is_cpu(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", None)
    assert resolve_providers() == ["CPUExecutionProvider"]


def test_model_device_reports_the_picked_device(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    from osam._models.sam3._models import Sam3

    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    fake_webgpu.append(types.SimpleNamespace(ep_name="WebGpuExecutionProvider"))
    assert Sam3().device == "webgpu"

    # onnxruntime falls back to CPU when WebGPU fails to start, e.g. no Vulkan.
    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort.InferenceSession, "get_providers", lambda self: ["CPUExecutionProvider"]
    )
    assert Sam3().device == "cpu"


def test_coreml_model_path_rewrites_gemm_to_trans_b(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["x", "w", "b"], ["y"]),
            helper.make_node("Gemm", ["x", "w", "b"], ["y1"], transB=0),
            helper.make_node("Gemm", ["x", "w_t", "b"], ["y2"], transB=1),
        ],
        "test",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4])],
        [
            helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            for name in ("y", "y1", "y2")
        ],
        [
            numpy_helper.from_array(np.arange(20, dtype=np.float32).reshape(4, 5), "w"),
            numpy_helper.from_array(
                np.arange(20, dtype=np.float32).reshape(5, 4), "w_t"
            ),
            numpy_helper.from_array(np.arange(5, dtype=np.float32), "b"),
        ],
    )
    gemm = Blob(url="https://example.com/gemm.onnx", hash="sha256:" + "0" * 64)
    other = Blob(url="https://example.com/other.onnx", hash="sha256:" + "1" * 64)

    def save(blob: Blob, graph: onnx.GraphProto) -> None:
        os.makedirs(os.path.dirname(blob.path), exist_ok=True)
        onnx.save(
            helper.make_model(
                graph, opset_imports=[helper.make_opsetid("", 21)], ir_version=10
            ),
            blob.path,
        )

    save(gemm, graph)
    save(
        other,
        helper.make_graph(
            [helper.make_node("Relu", ["x"], ["y"])],
            "test",
            [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3, 4])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, None)],
        ),
    )
    feeds = {"x": np.random.default_rng(0).random((3, 4), dtype=np.float32)}
    expected = onnxruntime.InferenceSession(gemm.path).run(None, feeds)

    path = _model._coreml_model_path(gemm)

    assert path != gemm.path
    actual = onnxruntime.InferenceSession(path).run(None, feeds)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    trans_b = [
        [a.i for a in node.attribute if a.name == "transB"]
        for node in onnx.load(path).graph.node
    ]
    assert trans_b == [[1], [1], [1]]
    # Nothing to rewrite: the original, read once.
    assert _model._coreml_model_path(other) == other.path
    assert os.path.islink(os.path.join(other.coreml_cache_path, "model.onnx"))
