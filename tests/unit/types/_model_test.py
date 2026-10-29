from typing import Any
from typing import Iterator

import onnxruntime
import pytest

from osam.types import _model
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
        def __init__(self, path: str, sess_options: Any, providers: Any) -> None:
            sessions.append((path, providers))
            self._providers = [p if isinstance(p, str) else p[0] for p in providers]

        def get_providers(self) -> list[str]:
            return self._providers

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
    fake_sessions: list[tuple[str, Any]],
) -> None:
    from osam._models.sam3._models import Sam3

    Sam3(device="coreml")

    coreml = [
        path
        for path, providers in fake_sessions
        if providers[0][0] == "CoreMLExecutionProvider"
    ]
    assert len(fake_sessions) == 3
    assert coreml == [Sam3._blobs["image_encoder"].path]


def test_device_unavailable_raises(fake_sessions: list[tuple[str, Any]]) -> None:
    from osam._models.sam3._models import Sam3

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
    with pytest.raises(RuntimeError, match="CoreMLExecutionProvider failed to load"):
        Sam3(device="coreml")
