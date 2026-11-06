from __future__ import annotations

import abc
import hashlib
import threading
from collections.abc import Callable

import imgviz
import numpy as np
import numpy.typing as npt
import onnxruntime
from loguru import logger

from .. import _onnx
from ..types import Annotation
from ..types import Blob
from ..types import BoundingBox
from ..types import Device
from ..types import ModelMetadata
from ._image_embedding import ImageEmbedding
from ._prompt import Prompt


class Model(abc.ABC):
    name: str
    metadata: ModelMetadata

    _blobs: dict[str, Blob]
    # Blobs that run on CoreML with device="coreml", all when None; the others
    # run on CPU.
    _coreml_blobs: frozenset[str] | None = None
    _inference_sessions: dict[str, onnxruntime.InferenceSession]

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
    def get_size(cls) -> int | None:
        size = 0
        for blob in cls._blobs.values():
            # Read once: a concurrent remove can turn a second read into None.
            blob_size: int | None = blob.size
            if blob_size is None:
                return None
            size += blob_size
        return size

    @classmethod
    def get_modified_at(cls) -> float | None:
        modified_at: float = 0
        for blob in cls._blobs.values():
            blob_modified_at: float | None = blob.modified_at
            if blob_modified_at is None:
                return None
            modified_at = max(modified_at, blob_modified_at)
        return modified_at

    @abc.abstractmethod
    def encode_image(self, *, image: np.ndarray) -> ImageEmbedding:
        # image: an RGB (H, W, 3) uint8 array.
        pass

    @classmethod
    @abc.abstractmethod
    def default_prompt(cls, *, height: int, width: int) -> Prompt:
        # The prompt used when a request gives none.
        pass

    def decode(
        self, *, embedding: ImageEmbedding, prompt: Prompt | None
    ) -> list[Annotation]:
        if prompt is None:
            prompt = self.default_prompt(
                height=embedding.original_height, width=embedding.original_width
            )
            logger.debug("Using the default prompt: {prompt!r}", prompt=prompt)
        return self._decode(embedding=embedding, prompt=prompt)

    @abc.abstractmethod
    def _decode(self, *, embedding: ImageEmbedding, prompt: Prompt) -> list[Annotation]:
        pass


def annotations_from_mask(
    *, mask: npt.NDArray[np.bool_], score: float | None = None
) -> list[Annotation]:
    # The mask cropped to its bounding box, or none for an empty mask.
    if not mask.any():
        return []
    ymin, xmin, ymax, xmax = imgviz.masks_to_bboxes(masks=[mask])[0].astype(int)
    return [
        Annotation(
            mask=mask[ymin : ymax + 1, xmin : xmax + 1],
            bounding_box=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
            score=score,
        )
    ]
