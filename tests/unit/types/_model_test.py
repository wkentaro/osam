import pathlib

import pytest

from osam.types._blob import Blob
from osam.types._model import Model
from osam.types._model import resolve_providers


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


def test_is_pulled_requires_every_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    class TwoBlobModel(Model):
        _blobs = {
            "encoder": Blob(url="https://example.com/encoder.onnx", hash="sha256:e"),
            "decoder": Blob(url="https://example.com/decoder.onnx", hash="sha256:d"),
        }

    pathlib.Path(TwoBlobModel._blobs["encoder"].path).parent.mkdir(parents=True)
    pathlib.Path(TwoBlobModel._blobs["encoder"].path).write_bytes(b"encoder")
    assert not TwoBlobModel.is_pulled()

    pathlib.Path(TwoBlobModel._blobs["decoder"].path).write_bytes(b"decoder")
    assert TwoBlobModel.is_pulled()
