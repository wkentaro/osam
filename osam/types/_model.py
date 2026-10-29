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

_PROVIDERS_ENV: Final = "OSAM_ONNX_PROVIDERS"


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


class Model(abc.ABC):
    name: str
    metadata: ModelMetadata

    _blobs: Dict[str, Blob]
    _inference_sessions: Dict[str, onnxruntime.InferenceSession]

    def __init__(self):
        logger.debug("Initializing model {name}", name=self.name)
        self.pull()
        self._inference_sessions = _load_inference_sessions(blobs=self._blobs)

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


def _load_inference_session(
    blob: Blob, providers: list[str] | None = None
) -> onnxruntime.InferenceSession:
    try:
        if providers is None:
            providers = resolve_providers()
        sess_options = onnxruntime.SessionOptions()  # type: ignore[possibly-missing-attribute]
        if "CUDAExecutionProvider" in providers and _setup_cuda():
            sess_options.add_session_config_entry("session.use_env_allocators", "1")
        inference_session = onnxruntime.InferenceSession(
            blob.path, sess_options, providers=providers
        )
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
