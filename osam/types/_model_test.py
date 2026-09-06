import pytest

from ._model import resolve_providers


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
