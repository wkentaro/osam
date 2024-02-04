import dataclasses
import hashlib
import os
from typing import Dict
from typing import Optional

import gdown
import numpy as np
import onnxruntime
from loguru import logger

from osam import types


@dataclasses.dataclass
class ModelBlob:
    url: str
    hash: str

    @property
    def path(self):
        return os.path.expanduser(f"~/.cache/osam/models/blobs/{self.hash}")

    @property
    def size(self):
        if os.path.exists(self.path):
            return os.stat(self.path).st_size
        else:
            return None

    @property
    def modified_at(self):
        if os.path.exists(self.path):
            return os.stat(self.path).st_mtime
        else:
            return None

    def pull(self):
        gdown.cached_download(url=self.url, path=self.path, hash=self.hash)

    def remove(self):
        if os.path.exists(self.path):
            logger.debug("Removing model blob {path!r}", path=self.path)
            os.remove(self.path)
        else:
            logger.warning("Model blob {path!r} not found", path=self.path)


class ModelBase:
    name: str

    _blobs: Dict[str, ModelBlob]
    _inference_sessions: Dict[str, onnxruntime.InferenceSession]

    def __init__(self):
        self.pull()
        self._inference_sessions = {
            key: onnxruntime.InferenceSession(blob.path)
            for key, blob in self._blobs.items()
        }

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

    def encode_image(self, image: np.ndarray) -> types.ImageEmbedding:
        raise NotImplementedError

    def generate_mask(
        self, image_embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> np.ndarray:
        raise NotImplementedError
