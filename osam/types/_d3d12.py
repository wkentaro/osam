import ctypes
import sys
import uuid
from typing import Any


class _Luid(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]


def _guid(*, value: str) -> ctypes.Array:
    return ctypes.create_string_buffer(uuid.UUID(value).bytes_le)


def _method(
    *, pointer: ctypes.c_void_p, index: int, restype: Any, argtypes: tuple = ()
) -> Any:
    if sys.platform != "win32":
        raise OSError("Direct3D 12 requires Windows")
    table = ctypes.cast(
        pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])


def _check_hresult(*, result: int) -> None:
    if result < 0:
        raise OSError(
            f"Direct3D 12 adapter probe failed: HRESULT {result & 0xFFFFFFFF:#x}"
        )


def supports_d3d12(*, luid: int) -> bool:
    # ONNX enumerates DXGI adapters independently of Dawn. A listed hardware
    # adapter can lack D3D12, causing Dawn to silently select the WARP CPU renderer.
    # Check that exact adapter by LUID, without creating a device or loading a model.
    if sys.platform != "win32":
        raise OSError("Direct3D 12 requires Windows")
    factory = ctypes.c_void_p()
    adapter = ctypes.c_void_p()
    try:
        create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
        create.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int32
        _check_hresult(
            result=create(
                _guid(value="1bc6ea02-ef36-464f-bf0c-21ca39e5168a"),  # IDXGIFactory4
                ctypes.byref(factory),
            )
        )
        enumerate_adapter = _method(
            pointer=factory,
            index=26,  # IDXGIFactory4::EnumAdapterByLuid
            restype=ctypes.c_int32,
            argtypes=(_Luid, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
        )
        _check_hresult(
            result=enumerate_adapter(
                factory,
                _Luid(luid & 0xFFFFFFFF, luid >> 32),
                _guid(value="29038f61-3839-4626-91fd-086879011a05"),  # IDXGIAdapter1
                ctypes.byref(adapter),
            )
        )
        check = ctypes.WinDLL("d3d12").D3D12CreateDevice
        check.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        check.restype = ctypes.c_int32
        # A null output checks support and returns S_FALSE (1) on success.
        return (
            check(
                adapter,
                0xB000,  # D3D_FEATURE_LEVEL_11_0, the minimum for D3D12
                _guid(value="189819f1-1db6-4b57-be54-1821339b85f7"),  # ID3D12Device
                None,
            )
            >= 0
        )
    finally:
        for pointer in (adapter, factory):
            if pointer:
                _method(pointer=pointer, index=2, restype=ctypes.c_ulong)(pointer)
