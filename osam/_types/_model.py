from __future__ import annotations

import abc
import hashlib
import threading
from collections.abc import Callable
from typing import Dict
from typing import Optional

import numpy as np
import onnxruntime
from loguru import logger

from .. import _onnx
from ..types import Blob
from ..types import Device
from ..types import ModelMetadata
from ._generate import GenerateRequest
from ._generate import GenerateResponse
from ._image_embedding import ImageEmbedding
from ._prompt import Prompt


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

        device: "cpu", "cuda", "coreml", or "webgpu". None picks OSAM_ONNX_PROVIDERS,
        or the first available of CUDA and a hardware WebGPU adapter, falling
        back to CPU.
        """
        logger.debug("Initializing model {name}", name=self.name)
        if device is not None:
            _onnx.device_providers(device=device)  # fail before downloading
        self.pull()
        if device is None:
            self._inference_sessions = _onnx.load_inference_sessions(blobs=self._blobs)
            return
        self._inference_sessions = {
            key: _onnx.create_device_session(
                blob=blob,
                providers=_onnx.device_providers(
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

    @property
    def device(self) -> str:
        # The device the model runs on, e.g. "webgpu", or "cpu" after a fallback.
        session = next(iter(self._inference_sessions.values()))
        provider: str = session.get_providers()[0]
        return {v: k for k, v in _onnx.DEVICE_PROVIDERS.items()}.get(provider, provider)

    @classmethod
    def pull(
        cls,
        *,
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
            # Read once: a concurrent remove can turn a second read into None.
            blob_size: Optional[int] = blob.size
            if blob_size is None:
                return None
            size += blob_size
        return size

    @classmethod
    def get_modified_at(cls) -> Optional[float]:
        modified_at: float = 0
        for blob in cls._blobs.values():
            blob_modified_at: Optional[float] = blob.modified_at
            if blob_modified_at is None:
                return None
            modified_at = max(modified_at, blob_modified_at)
        return modified_at

    def encode_image(self, *, image: np.ndarray) -> ImageEmbedding:
        raise NotImplementedError

    @classmethod
    def default_prompt(cls, *, height: int, width: int) -> Prompt:
        # The prompt used when a request gives none.
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, *, request: GenerateRequest) -> GenerateResponse:
        pass
