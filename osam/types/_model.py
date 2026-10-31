from __future__ import annotations

import abc
import contextlib
import functools
import hashlib
import os
import sys
import threading
from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import Final
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import cast

import numpy as np
import onnxruntime
from loguru import logger

from ._blob import Blob
from ._generate import GenerateRequest
from ._generate import GenerateResponse
from ._image_embedding import ImageEmbedding
from ._model_metadata import ModelMetadata
from ._prompt import Prompt

_PROVIDERS_ENV: Final = "OSAM_ONNX_PROVIDERS"

Device = Literal["cpu", "cuda", "coreml", "webgpu"]
_Providers = Sequence[str | tuple[str, dict[str, str]]]

_DEVICE_PROVIDERS: Final[dict[str, str]] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
}
# MLProgram supports more operators than the default NeuralNetwork, and leaving
# out the Neural Engine keeps the GPU's accuracy and speed.
_COREML_OPTIONS: Final[dict[str, str]] = {
    "ModelFormat": "MLProgram",
    "MLComputeUnits": "CPUAndGPU",
}


def resolve_providers() -> list[str]:
    raw = os.environ.get(_PROVIDERS_ENV, "")
    providers = [entry.strip() for entry in raw.split(",") if entry.strip()]
    if providers:
        return providers
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():  # type: ignore[possibly-missing-attribute]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@functools.cache
def _setup_cuda() -> bool:
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():  # type: ignore[possibly-missing-attribute]
        return False
    ort = cast(Any, onnxruntime)  # ty cannot see these members
    try:
        # Load the CUDA libraries from the nvidia-* wheels of osam[gpu]. It
        # prints failures to stdout, which would break `osam run --json`.
        with contextlib.redirect_stdout(sys.stderr):
            ort.preload_dlls()
        # Share one CUDA arena across sessions. With an arena per session,
        # SAM3's three sessions run out of memory on a 12 GB GPU.
        ort.create_and_register_allocator_v2(
            "CUDAExecutionProvider",
            ort.OrtMemoryInfo(
                "Cuda",
                ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR,
                0,
                ort.OrtMemType.DEFAULT,
            ),
            {},
            ort.OrtArenaCfg({"arena_extend_strategy": 1}),  # kSameAsRequested
        )
    except Exception as e:
        logger.warning("Failed to set up a shared CUDA arena: {e}", e=e)
        return False
    return True


@functools.cache
def _webgpu_device() -> Any | None:
    try:
        import onnxruntime_ep_webgpu as webgpu_ep  # type: ignore[unresolved-import]
    except ImportError:
        raise RuntimeError(
            "WebGpuExecutionProvider needs onnxruntime-ep-webgpu: "
            "pip install 'osam[webgpu]'"
        ) from None
    ort = cast(Any, onnxruntime)
    ort.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())
    # The plugin accepts one device.
    for device in ort.get_ep_devices():
        if device.ep_name == webgpu_ep.get_ep_name():
            return device
    return None


class Model(abc.ABC):
    name: str
    metadata: ModelMetadata

    _blobs: Dict[str, Blob]
    # Blobs that run on CoreML with device="coreml", all when None; the others
    # run on CPU.
    _coreml_blobs: Optional[frozenset[str]] = None
    _inference_sessions: Dict[str, onnxruntime.InferenceSession]

    def __init__(self, device: Device | None = None):
        """Load the model.

        device: "cpu", "cuda", "coreml", or "webgpu". None picks OSAM_ONNX_PROVIDERS, or
        CUDA when available, falling back to CPU.
        """
        logger.debug("Initializing model {name}", name=self.name)
        if device is not None:
            _device_providers(device=device)  # fail before downloading
        self.pull()
        if device is None:
            self._inference_sessions = _load_inference_sessions(blobs=self._blobs)
            return
        self._inference_sessions = {
            key: _create_device_session(
                blob=blob,
                providers=_device_providers(
                    device=device,
                    coreml=self._coreml_blobs is None or key in self._coreml_blobs,
                    coreml_cache_path=blob.coreml_cache_path,
                ),
            )
            for key, blob in self._blobs.items()
        }
        logger.info(
            "Initialized inference sessions on {device!r}: {providers!r}",
            device=device,
            providers={
                k: s.get_providers()[0] for k, s in self._inference_sessions.items()
            },
        )

    @classmethod
    def pull(
        cls,
        progress: Callable[[str, int, int | None], None] | None = None,
        cancel: threading.Event | None = None,
        timeout: float | tuple[float, float] | None = 30,
    ) -> None:
        for blob in cls._blobs.values():
            blob.pull(progress=progress, cancel=cancel, timeout=timeout)

    @classmethod
    def remove(cls):
        for blob in cls._blobs.values():
            blob.remove()

    @classmethod
    def get_id(cls) -> str:
        return hashlib.md5(
            "+".join(blob.hash for blob in cls._blobs.values()).encode()
        ).hexdigest()[:12]

    @classmethod
    def get_size(cls) -> Optional[int]:
        size = 0
        for blob in cls._blobs.values():
            if blob.size is None:
                return None
            size += blob.size
        return size

    @classmethod
    def get_modified_at(cls) -> Optional[float]:
        modified_at: float = 0
        for blob in cls._blobs.values():
            if blob.modified_at is None:
                return None
            modified_at = max(modified_at, blob.modified_at)
        return modified_at

    def encode_image(self, image: np.ndarray) -> ImageEmbedding:
        raise NotImplementedError

    def default_prompt(self, height: int, width: int) -> Prompt:
        """Return the prompt used when a request gives none."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, request: GenerateRequest) -> GenerateResponse:
        pass


def _load_inference_sessions(
    blobs: dict[str, Blob],
) -> dict[str, onnxruntime.InferenceSession]:
    providers: Sequence[str] | None = None
    inference_sessions: dict[str, onnxruntime.InferenceSession] = {}
    for key, blob in blobs.items():
        inference_session: onnxruntime.InferenceSession = _load_inference_session(
            blob=blob, providers=providers
        )
        providers = inference_session.get_providers()
        inference_sessions[key] = inference_session
    logger.info(
        "Initialized inference sessions with providers {providers!r}",
        providers=providers,
    )
    return inference_sessions


def _device_providers(
    device: str, coreml: bool = True, coreml_cache_path: str | None = None
) -> _Providers:
    if device not in _DEVICE_PROVIDERS:
        raise ValueError(
            f"device must be one of {', '.join(_DEVICE_PROVIDERS)}: {device!r}"
        )
    provider: str = _DEVICE_PROVIDERS[device]
    if device == "webgpu":
        if _webgpu_device() is None:
            raise RuntimeError(f"{provider} found no GPU")
    elif provider not in onnxruntime.get_available_providers():  # type: ignore[possibly-missing-attribute]
        raise RuntimeError(f"{provider} is unavailable in the installed onnxruntime")
    if device == "cpu" or (device == "coreml" and not coreml):
        return ["CPUExecutionProvider"]
    if device == "coreml":
        options: dict[str, str] = dict(_COREML_OPTIONS)
        if coreml_cache_path is not None:
            # Compiling takes seconds per model; reuse it on the next load.
            # Models compiled by another onnxruntime version fail to run.
            options["ModelCacheDirectory"] = os.path.join(
                coreml_cache_path, onnxruntime.__version__
            )
        return [(provider, options), "CPUExecutionProvider"]
    return [provider, "CPUExecutionProvider"]


def _create_session(blob: Blob, providers: _Providers) -> onnxruntime.InferenceSession:
    sess_options = onnxruntime.SessionOptions()  # type: ignore[possibly-missing-attribute]
    if "CUDAExecutionProvider" in providers and _setup_cuda():
        sess_options.add_session_config_entry("session.use_env_allocators", "1")
    if providers[0] == "WebGpuExecutionProvider":
        # A plugin provider is added by device; CPU stays as the fallback.
        sess_options.add_provider_for_devices([_webgpu_device()], {})
        return onnxruntime.InferenceSession(blob.path, sess_options)
    return onnxruntime.InferenceSession(blob.path, sess_options, providers=providers)


def _create_device_session(
    blob: Blob, providers: _Providers
) -> onnxruntime.InferenceSession:
    session: onnxruntime.InferenceSession = _create_session(
        blob=blob, providers=providers
    )
    # onnxruntime silently drops a provider it can't use.
    provider: str = providers[0] if isinstance(providers[0], str) else providers[0][0]
    if provider not in session.get_providers():
        raise RuntimeError(f"{provider} failed to load {blob.path}")
    # Raise on a provider failure in run() too, instead of retrying on CPU.
    session.disable_fallback()
    return session


def _load_inference_session(
    blob: Blob, providers: list[str] | None = None
) -> onnxruntime.InferenceSession:
    try:
        if providers is None:
            providers = resolve_providers()
        inference_session = _create_session(blob=blob, providers=providers)
    except Exception as e:
        # Even though there is fallback in onnxruntime, it won't always work.
        # e.g., CUDA is installed and CUDA_PATH is set, but CUDA_VISIBLE_DEVICES
        # is empty. We fallback to cpu in such cases.
        logger.error(
            "Failed to create inference session with providers {providers!r}. "
            "Falling back to ['CPUExecutionProvider']: {e}",
            providers=providers,
            e=e,
        )
        providers = ["CPUExecutionProvider"]
        inference_session = onnxruntime.InferenceSession(blob.path, providers=providers)
    return inference_session
