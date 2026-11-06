import ctypes
import types
import uuid
from typing import Any

import pytest

from osam._types import _d3d12


@pytest.mark.parametrize(
    ("factory_result", "adapter_result", "device_result", "expected", "released"),
    [
        (0, 0, 1, True, [20, 10]),  # S_FALSE means the capability check succeeded.
        (0, 0, -2005270524, False, [20, 10]),  # DXGI_ERROR_UNSUPPORTED
        (-1, 0, 1, None, []),
        (0, -1, 1, None, [10]),
        (0, 0, OSError("Missing d3d12.dll"), None, [20, 10]),
    ],
)
def test_probe_checks_exact_adapter_and_releases_interfaces(
    monkeypatch: pytest.MonkeyPatch,
    factory_result: int,
    adapter_result: int,
    device_result: int | OSError,
    expected: bool | None,
    released: list[int],
) -> None:
    monkeypatch.setattr(_d3d12.sys, "platform", "win32")
    releases: list[int] = []
    calls: list[str] = []

    def guid_bytes(pointer: Any) -> bytes:
        return ctypes.string_at(pointer, 16)

    def factory(iid: Any, output: Any) -> int:
        assert (
            guid_bytes(iid)
            == uuid.UUID("1bc6ea02-ef36-464f-bf0c-21ca39e5168a").bytes_le
        )
        if factory_result >= 0:
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 10
        return factory_result

    def adapter(pointer: Any, luid: Any, iid: Any, output: Any) -> int:
        assert pointer.value == 10
        assert luid.low == 0xAABBCCDD
        assert luid.high & 0xFFFFFFFF == 0xFEDCBA98
        assert (
            guid_bytes(iid)
            == uuid.UUID("29038f61-3839-4626-91fd-086879011a05").bytes_le
        )
        if adapter_result >= 0:
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 20
        return adapter_result

    def check(pointer: Any, feature_level: int, iid: Any, output: Any) -> int:
        assert pointer.value == 20
        assert feature_level == 0xB000
        assert (
            guid_bytes(iid)
            == uuid.UUID("189819f1-1db6-4b57-be54-1821339b85f7").bytes_le
        )
        assert output is None  # Probe support without allocating a device.
        calls.append("check")
        assert isinstance(device_result, int)
        return device_result

    def dll(name: str) -> Any:
        if name == "dxgi":
            return types.SimpleNamespace(CreateDXGIFactory1=factory)
        assert name == "d3d12"
        if isinstance(device_result, OSError):
            raise device_result
        return types.SimpleNamespace(D3D12CreateDevice=check)

    def method(*, pointer: Any, index: int, restype: Any, argtypes: tuple = ()) -> Any:
        if index == 26:
            return adapter
        assert index == 2
        return lambda pointer: releases.append(pointer.value)

    monkeypatch.setattr(ctypes, "WinDLL", dll, raising=False)
    monkeypatch.setattr(_d3d12, "_method", method)
    if expected is None:
        with pytest.raises(OSError):
            _d3d12.supports_d3d12(luid=0xFEDCBA98AABBCCDD)
    else:
        assert _d3d12.supports_d3d12(luid=0xFEDCBA98AABBCCDD) is expected
        assert calls == ["check"]
    assert releases == released
