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
