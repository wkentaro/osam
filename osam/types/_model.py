from __future__ import annotations

import abc
import hashlib
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import onnxruntime
from loguru import logger

from ._blob import Blob
from ._generate import GenerateRequest
from ._generate import GenerateResponse
from ._image_embedding import ImageEmbedding


class Model(abc.ABC):
    name: str

    _blobs: Dict[str, Blob]
    _inference_sessions: Dict[str, onnxruntime.InferenceSession]

    def __init__(self):
        logger.debug("Initializing model {name}", name=self.name)
        self.pull()
        self._inference_sessions = _load_inference_sessions(blobs=self._blobs)

    @classmethod
    def pull(cls):
        for blob in cls._blobs.values():
            blob.pull()

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
        # Try to use all of the available providers e.g., cuda, tensorrt.
        if providers is None:
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():  # type: ignore[possibly-missing-attribute]
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        inference_session = onnxruntime.InferenceSession(blob.path, providers=providers)
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
