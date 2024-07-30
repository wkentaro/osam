import hashlib
from typing import Dict
from typing import Optional

import numpy as np
import onnxruntime
from loguru import logger

from .. import _contextlib
from ._blob import Blob
from ._generate import GenerateRequest
from ._generate import GenerateResponse
from ._image_embedding import ImageEmbedding


class Model:
    name: str

    _blobs: Dict[str, Blob]
    _inference_sessions: Dict[str, onnxruntime.InferenceSession]

    def __init__(self):
        logger.debug("Initializing model {name}", name=self.name)

        self.pull()

        providers = None
        self._inference_sessions = {}
        for key, blob in self._blobs.items():
            try:
                # Try to use all of the available providers e.g., cuda, tensorrt.
                if providers is None:
                    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    else:
                        providers = ["CPUExecutionProvider"]
                # Suppress all the error messages from the missing providers.
                with _contextlib.suppress():
                    inference_session = onnxruntime.InferenceSession(
                        blob.path, providers=providers
                    )
            except Exception as e:
                # Even though there is fallback in onnxruntime, it won't always work.
                # e.g., CUDA is installed and CUDA_PATH is set, but CUDA_VISIBLE_DEVICES
                # is empty. We fallback to cpu in such cases.
                logger.error(
                    "Failed to create inference session with providers {providers!r}. "
                    "Falling back to ['CPUExecutionProvider']",
                    providers=providers,
                    e=e,
                )
                providers = ["CPUExecutionProvider"]
                inference_session = onnxruntime.InferenceSession(
                    blob.path, providers=providers
                )
            self._inference_sessions[key] = inference_session

            providers = inference_session.get_providers()
        logger.info(
            "Initialized inference sessions with providers {providers!r}",
            providers=providers,
        )

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
        modified_at = 0
        for blob in cls._blobs.values():
            if blob.modified_at is None:
                return None
            modified_at = max(modified_at, blob.modified_at)
        return modified_at

    def encode_image(self, image: np.ndarray) -> ImageEmbedding:
        raise NotImplementedError

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError
