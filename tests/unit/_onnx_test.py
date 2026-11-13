import os
import pathlib
import sys
import types
from collections.abc import Iterator
from typing import Any
from typing import cast

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import helper
from onnx import numpy_helper

from osam import _errors
from osam import _onnx
from osam._onnx import resolve_providers
from osam._types import _model
from osam.types._blob import Blob


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
    _onnx._setup_cuda.cache_clear()
    yield calls
    _onnx._setup_cuda.cache_clear()


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
    assert _onnx._setup_cuda() is False
    assert fake_cuda == []


def test_setup_cuda_registers_shared_arena_once(fake_cuda: list[str]) -> None:
    assert _onnx._setup_cuda() is True
    assert _onnx._setup_cuda() is True
    assert fake_cuda == ["preload", "register"]


def test_setup_cuda_keeps_preload_output_off_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_cuda: list[str],
) -> None:
    ort: Any = onnxruntime
    monkeypatch.setattr(ort, "preload_dlls", lambda: print("Failed to load libcudnn"))
    _onnx._setup_cuda()
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

    _onnx._load_inference_session(blob=blob, providers=providers)

    [(sess_options, used_providers)] = sessions
    assert used_providers == providers
    with pytest.raises(RuntimeError, match="does not have configuration"):
        sess_options.get_session_config_entry("session.use_env_allocators")


@pytest.fixture
def fake_sessions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    sessions: list[tuple[str, Any]] = []

    class FakeSession:
        _providers: list[str]

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
    monkeypatch.setattr(_model.Model, "pull", classmethod(lambda cls, **kwargs: None))
    return sessions


def test_device_coreml_runs_only_coreml_blobs_on_coreml(
    monkeypatch: pytest.MonkeyPatch, fake_sessions: list[tuple[str, Any]]
) -> None:
    from osam._models.sam3._models import Sam3

    monkeypatch.setattr(_onnx, "_coreml_model", lambda path: f"{path}.t")
    monkeypatch.setattr(_onnx, "_webgpu_device", lambda: None)
    Sam3(device="coreml")

    blobs = Sam3._device_blobs(device="coreml")
    coreml = [
        (path, providers[0][1]["ModelCacheDirectory"])
        for path, providers in fake_sessions
        if providers[0][0] == "CoreMLExecutionProvider"
    ]
    assert coreml == [
        (
            f"{blobs[key].path}.t",
            os.path.join(blobs[key].coreml_cache_path, onnxruntime.__version__),
        )
        for key in ("image_encoder", "memory_attention_fixed")
    ]
    # Without a WebGPU adapter, the others run on the CPU.
    disabled = [p for p, providers in fake_sessions if providers == "fallback disabled"]
    assert disabled == [
        "CoreMLExecutionProvider",
        *["CPUExecutionProvider"] * 6,
        "CoreMLExecutionProvider",
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
    monkeypatch.setattr(_onnx, "_coreml_model", lambda path: path)
    with pytest.raises(RuntimeError, match="CoreMLExecutionProvider failed to load"):
        Sam3(device="coreml")


def _webgpu_adapter(
    *, vendor_id: int = 0x10DE, device_id: int = 0x2208, description: str = ""
) -> Any:
    metadata: dict[str, str] = {"LUID": "1", "Description": description}
    return types.SimpleNamespace(
        ep_name="WebGpuExecutionProvider",
        device=types.SimpleNamespace(
            vendor="", vendor_id=vendor_id, device_id=device_id, metadata=metadata
        ),
    )


def _warp_adapter() -> Any:
    return _webgpu_adapter(
        vendor_id=0x1414, device_id=0x008C, description="Microsoft Basic Render Driver"
    )


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
        devices: Any

        def add_provider_for_devices(self, devices: Any, options: Any) -> None:
            self.devices = devices

    ort: Any = onnxruntime
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", plugin)
    monkeypatch.setattr(ort, "register_execution_provider_library", lambda *a: None)
    monkeypatch.setattr(ort, "get_ep_devices", lambda: devices)
    monkeypatch.setattr(ort, "SessionOptions", FakeSessionOptions)
    monkeypatch.setattr(_onnx, "supports_d3d12", lambda *, luid: True)
    _onnx._webgpu_device.cache_clear()
    yield devices
    _onnx._webgpu_device.cache_clear()


def test_device_webgpu_runs_every_blob_on_webgpu(
    fake_webgpu: list[Any], fake_sessions: list[tuple[str, Any]]
) -> None:
    from osam._models.sam3._models import Sam3

    fake_webgpu.append(_webgpu_adapter())
    Sam3(device="webgpu")

    disabled = [p for p, providers in fake_sessions if providers == "fallback disabled"]
    assert disabled == ["WebGpuExecutionProvider"] * len(
        Sam3._device_blobs(device="webgpu")
    )


def test_device_webgpu_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    from osam._models.sam3._models import Sam3

    with pytest.raises(RuntimeError, match="WebGpuExecutionProvider found no GPU"):
        Sam3(device="webgpu")

    monkeypatch.setitem(sys.modules, "onnxruntime_ep_webgpu", None)
    _onnx._webgpu_device.cache_clear()
    with pytest.raises(RuntimeError, match=r"pip install 'osam\[webgpu\]'"):
        Sam3(device="webgpu")


def test_resolve_providers_picks_webgpu_after_cuda(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    assert resolve_providers() == ["CPUExecutionProvider"]  # no GPU listed

    fake_webgpu.append(_webgpu_adapter())
    _onnx._webgpu_device.cache_clear()
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


def test_sam3_runs_on_cpu_with_cuda_before_onnxruntime_1_24(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda: list[str],
    fake_sessions: list[tuple[str, Any]],
) -> None:
    from osam._models.sam3 import _models

    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(_models, "_ORT_VERSION", (1, 23))
    monkeypatch.setattr(_model.Model, "pull", classmethod(lambda cls, **kwargs: None))
    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    with pytest.raises(RuntimeError, match="fails on cuda with onnxruntime"):
        _models.Sam3(device="cuda")
    # Picking a device automatically picks the CPU, with its files.
    sam3 = _models.Sam3()
    assert sam3.device == "cpu"
    cpu = _models.Sam3._device_blobs(device="cpu")
    assert _models.Sam3._blobs_of(device=None) == cpu
    loaded = [path for path, p in fake_sessions if p != "fallback disabled"]
    assert loaded == [blob.path for blob in cpu.values()]

    monkeypatch.setattr(_models, "_ORT_VERSION", (1, 24))
    assert _models.Sam3(device="cuda").device == "cuda"


def test_model_device_reports_the_picked_device(
    monkeypatch: pytest.MonkeyPatch,
    fake_webgpu: list[Any],
    fake_sessions: list[tuple[str, Any]],
) -> None:
    from osam._models.sam3._models import Sam3

    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(Sam3, "_runs_faster_on_coreml", classmethod(lambda cls: False))
    fake_webgpu.append(_webgpu_adapter())
    assert Sam3().device == "webgpu"

    # onnxruntime falls back to CPU when WebGPU fails to start, e.g. no Vulkan.
    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort.InferenceSession, "get_providers", lambda self: ["CPUExecutionProvider"]
    )
    fake_sessions.clear()
    assert Sam3().device == "cpu"
    # It reloads with the files that run fastest on the CPU.
    cpu = [blob.path for blob in Sam3._device_blobs(device="cpu").values()]
    loaded = [path for path, p in fake_sessions if p != "fallback disabled"]
    assert loaded[-len(cpu) :] == cpu


def test_cpu_fallback_without_auto_pull_raises_for_the_cpu_files(
    monkeypatch: pytest.MonkeyPatch,
    fake_webgpu: list[Any],
    fake_sessions: list[tuple[str, Any]],
) -> None:
    from osam._models.sam3._models import Sam3

    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(Sam3, "_runs_faster_on_coreml", classmethod(lambda cls: False))
    fake_webgpu.append(_webgpu_adapter())
    ort: Any = onnxruntime
    monkeypatch.setattr(
        ort.InferenceSession, "get_providers", lambda self: ["CPUExecutionProvider"]
    )
    # Only the WebGPU files are on disk.
    cpu = {blob.path for blob in Sam3._device_blobs(device="cpu").values()}
    monkeypatch.setattr(Blob, "verified", lambda self: self.path not in cpu)
    with pytest.raises(_errors.NotPulledError) as e:
        Sam3(auto_pull=False)
    assert e.value.device == "cpu"


def test_partial_fallback_moves_every_session_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Session:
        _providers: list[str]

        def __init__(self, path: str, providers: list[str]) -> None:
            self._providers = providers

        def get_providers(self) -> list[str]:
            return self._providers

    # The decoder runs out of GPU memory and falls back to CPU alone.
    cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    cpu = ["CPUExecutionProvider"]
    loaded = iter([Session("", cuda), Session("", cpu)])

    def load(*, blob: Any, providers: list[str] | None) -> Session:
        return Session("", cpu) if providers == cpu else next(loaded)

    monkeypatch.setattr(_onnx, "_load_inference_session", load)
    blobs = cast(Any, {key: types.SimpleNamespace(path=key) for key in "abc"})
    sessions = _onnx.load_inference_sessions(blobs=blobs)
    assert {k: s.get_providers() for k, s in sessions.items()} == dict.fromkeys(
        "abc", cpu
    )


def test_sam3_picks_coreml_on_macos(
    monkeypatch: pytest.MonkeyPatch,
    fake_webgpu: list[Any],
    fake_sessions: list[tuple[str, Any]],
) -> None:
    from osam._models.sam3 import _models

    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(_onnx, "_coreml_model", lambda path: path)
    monkeypatch.setattr(_models, "_ORT_VERSION", (1, 30))
    fake_webgpu.append(_webgpu_adapter())
    # The image encoder on CoreML, as with device="coreml", and the rest on
    # WebGPU.
    sam3 = _models.Sam3()
    assert sam3.device == "coreml"
    assert {k: s.get_providers()[0] for k, s in sam3._inference_sessions.items()} == {
        k: "CoreMLExecutionProvider"
        if k in _models.Sam3._coreml_blobs
        else "WebGpuExecutionProvider"
        for k in _models.Sam3._device_blobs(device="coreml")
    }

    # Not when CoreML fails to load it.
    def fail(**kwargs: Any) -> None:
        raise RuntimeError("CoreMLExecutionProvider failed")

    with monkeypatch.context() as m:
        m.setattr(_onnx, "create_device_session", fail)
        assert _models.Sam3().device == "webgpu"
    # Nor with providers named, nor onnxruntime before 1.24.
    with monkeypatch.context() as m:
        m.setenv("OSAM_ONNX_PROVIDERS", "WebGpuExecutionProvider,CPUExecutionProvider")
        assert _models.Sam3().device == "webgpu"
        m.setenv("OSAM_ONNX_PROVIDERS", "CoreMLExecutionProvider,CPUExecutionProvider")
        providers = {
            s.get_providers()[0] for s in _models.Sam3()._inference_sessions.values()
        }
        assert "WebGpuExecutionProvider" not in providers
    monkeypatch.setattr(_models, "_ORT_VERSION", (1, 23))
    assert _models.Sam3().device == "webgpu"


def test_sam3_picks_the_files_on_disk_when_coreml_files_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osam._models.sam3 import _models

    monkeypatch.setattr(_onnx, "picks_coreml", lambda: True)
    monkeypatch.setattr(_onnx, "default_device", lambda: "webgpu")
    monkeypatch.setattr(_models, "_ORT_VERSION", (1, 30))
    coreml = _models.Sam3._device_blobs(device="coreml")
    webgpu = _models.Sam3._device_blobs(device="webgpu")
    on_disk: list[Blob] = []
    monkeypatch.setattr(Blob, "size", property(lambda b: 1 if b in on_disk else None))

    assert _models.Sam3._blobs_of(device=None) == coreml
    on_disk.extend(webgpu.values())
    assert _models.Sam3._blobs_of(device=None) == webgpu
    # But a pull fetches the files that loading tries first.
    pulled: list[Blob] = []
    monkeypatch.setattr(Blob, "pull", lambda b, **kwargs: pulled.append(b))
    _models.Sam3.pull()
    assert pulled == list(coreml.values())
    on_disk.extend(coreml.values())
    assert _models.Sam3._blobs_of(device=None) == coreml


def test_coreml_model_rewrites_gemm_to_trans_b(
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

    model = _onnx._coreml_model(path=gemm.path)

    assert isinstance(model, bytes)
    actual = onnxruntime.InferenceSession(model).run(None, feeds)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    trans_b = [
        [a.i for a in node.attribute if a.name == "transB"]
        for node in onnx.load_from_string(model).graph.node
    ]
    assert trans_b == [[1], [1], [1]]
    assert _onnx._coreml_model(path=other.path) == other.path
    assert not os.path.exists(gemm.coreml_cache_path)

    # The session drops the bytes and falls back from the original file.
    monkeypatch.setattr(
        _onnx, "_first_provider", lambda providers: "CoreMLExecutionProvider"
    )
    session = _onnx._create_session(path=gemm.path, providers=["CPUExecutionProvider"])
    assert session._model_bytes is None
    session.set_providers(["CPUExecutionProvider"])
    for a, e in zip(session.run(None, feeds), expected):
        np.testing.assert_array_equal(a, e)


def test_resolve_providers_skips_software_webgpu(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    fake_webgpu.append(_warp_adapter())
    assert resolve_providers() == ["CPUExecutionProvider"]

    # A GPU without its driver installed renders on WARP too.
    fake_webgpu[1:] = [
        _webgpu_adapter(
            vendor_id=0x1234,
            device_id=0x1111,
            description="Microsoft Basic Display Adapter",
        )
    ]
    _onnx._webgpu_device.cache_clear()
    assert resolve_providers() == ["CPUExecutionProvider"]


def test_resolve_providers_picks_hardware_webgpu_after_software(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    hardware = _webgpu_adapter()
    fake_webgpu += [_warp_adapter(), hardware]
    assert resolve_providers() == ["WebGpuExecutionProvider", "CPUExecutionProvider"]
    assert _onnx._webgpu_device() is hardware


@pytest.mark.parametrize(
    "adapter",
    [
        # An integrated GPU shares system memory: no dedicated VRAM.
        _webgpu_adapter(
            vendor_id=0x8086,
            device_id=0x46A6,
            description="Intel(R) Iris(R) Xe Graphics",
        ),
        # A VM with a passed-through GPU, or a Microsoft-made GPU driver.
        _webgpu_adapter(
            vendor_id=0x1414, device_id=0x0001, description="Microsoft Hyper-V Video"
        ),
    ],
)
def test_resolve_providers_keeps_hardware_webgpu(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any], adapter: Any
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    fake_webgpu.append(adapter)
    assert resolve_providers() == ["WebGpuExecutionProvider", "CPUExecutionProvider"]


def test_device_webgpu_runs_on_software_when_requested(
    fake_webgpu: list[Any], fake_sessions: list[tuple[str, Any]]
) -> None:
    from osam._models.sam3._models import Sam3

    fake_webgpu.append(_warp_adapter())
    Sam3(device="webgpu")

    disabled = [p for p, providers in fake_sessions if providers == "fallback disabled"]
    assert disabled == ["WebGpuExecutionProvider"] * len(
        Sam3._device_blobs(device="webgpu")
    )


@pytest.mark.parametrize("supported", [False, True])
def test_windows_webgpu_requires_d3d12_hardware(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any], supported: bool
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    checked: list[int] = []

    def check(*, luid: int) -> bool:
        checked.append(luid)
        return supported

    monkeypatch.setattr(_onnx, "supports_d3d12", check)
    fake_webgpu.append(
        _webgpu_adapter(
            vendor_id=0x15AD, device_id=0x0406, description="VMware SVGA 3D"
        )
    )
    assert resolve_providers() == (
        ["WebGpuExecutionProvider", "CPUExecutionProvider"]
        if supported
        else ["CPUExecutionProvider"]
    )
    assert checked and set(checked) == {1}


def test_windows_webgpu_prefers_d3d12_capable_adapter(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_onnx, "supports_d3d12", lambda *, luid: luid == 2)
    unsupported = _webgpu_adapter(description="VMware SVGA 3D")
    supported = _webgpu_adapter(description="Passed-through GPU")
    supported.device.metadata["LUID"] = "2"
    fake_webgpu += [unsupported, supported]
    assert resolve_providers() == ["WebGpuExecutionProvider", "CPUExecutionProvider"]
    assert _onnx._webgpu_device() is supported


@pytest.mark.parametrize("metadata", [{}, {"LUID": "invalid"}])
def test_windows_webgpu_missing_adapter_identity_uses_cpu(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any], metadata: dict[str, str]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    adapter = _webgpu_adapter()
    adapter.device.metadata = metadata
    fake_webgpu.append(adapter)
    assert resolve_providers() == ["CPUExecutionProvider"]


def test_windows_webgpu_probe_failure_uses_cpu(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any]
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    def check(*, luid: int) -> bool:
        raise OSError("Cannot load d3d12.dll")

    monkeypatch.setattr(_onnx, "supports_d3d12", check)
    fake_webgpu.append(_webgpu_adapter())
    assert resolve_providers() == ["CPUExecutionProvider"]


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_non_windows_webgpu_does_not_probe_d3d12(
    monkeypatch: pytest.MonkeyPatch, fake_webgpu: list[Any], platform: str
) -> None:
    monkeypatch.delenv("OSAM_ONNX_PROVIDERS", raising=False)
    monkeypatch.setattr(sys, "platform", platform)

    def check(*, luid: int) -> bool:
        pytest.fail("Direct3D 12 is only checked on Windows")

    monkeypatch.setattr(_onnx, "supports_d3d12", check)
    fake_webgpu.append(_webgpu_adapter())
    assert resolve_providers() == ["WebGpuExecutionProvider", "CPUExecutionProvider"]


def test_windows_unsupported_hardware_can_be_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch,
    fake_webgpu: list[Any],
    fake_sessions: list[tuple[str, Any]],
) -> None:
    from osam._models.sam3._models import Sam3

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(_onnx, "supports_d3d12", lambda *, luid: False)
    fake_webgpu.append(_webgpu_adapter(description="VMware SVGA 3D"))
    Sam3(device="webgpu")
    assert [
        p for p, providers in fake_sessions if providers == "fallback disabled"
    ] == ["WebGpuExecutionProvider"] * len(Sam3._device_blobs(device="webgpu"))
    monkeypatch.setenv("OSAM_ONNX_PROVIDERS", "WebGpuExecutionProvider")
    assert resolve_providers() == ["WebGpuExecutionProvider"]


def test_run_on_device_keeps_gpu_outputs_on_the_gpu() -> None:
    from unittest import mock

    import numpy as np

    from osam import _types as types

    image = np.zeros(1, np.float32)
    session = mock.Mock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    session.run.return_value = [np.ones(1)]
    assert _onnx.run_on_device(session=session, names=["a"], inputs={"x": image}) == [
        np.ones(1)
    ]
    session.io_binding.assert_not_called()

    # On a GPU, the outputs stay there, and OrtValue inputs bind as they are.
    feature = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((2, 3), np.float32))
    session.get_providers.return_value = ["CUDAExecutionProvider"]
    binding = session.io_binding.return_value
    _onnx.run_on_device(session=session, names=["a"], inputs={"x": image, "f": feature})
    binding.bind_cpu_input.assert_called_once_with("x", image)
    binding.bind_ortvalue_input.assert_called_once_with("f", feature)
    binding.bind_output.assert_called_once_with("a", "cuda")

    # The embedding cache counts an OrtValue's bytes as an array's.
    embedding = types.ImageEmbedding(
        original_height=1, original_width=1, embedding=feature
    )
    assert embedding.nbytes == 24
