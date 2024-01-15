import hashlib
import os
from typing import Optional

import gdown
import onnxruntime


class SamBase:
    _encoder_path: str
    _encoder_md5: str
    _encoder_url: Optional[str]

    _decoder_path: str
    _decoder_md5: str
    _decoder_url: Optional[str]

    _encoder_session: onnxruntime.InferenceSession
    _decoder_session: onnxruntime.InferenceSession

    def __init__(self):
        gdown.cached_download(
            url=self._encoder_url, md5=self._encoder_md5, path=self._encoder_path
        )
        gdown.cached_download(
            url=self._decoder_url, md5=self._decoder_md5, path=self._decoder_path
        )
        self._encoder_session = onnxruntime.InferenceSession(self._encoder_path)
        self._decoder_session = onnxruntime.InferenceSession(self._decoder_path)

    @classmethod
    def get_id(cls):
        return hashlib.md5((cls._encoder_md5 + cls._decoder_md5).encode()).hexdigest()[
            :12
        ]

    @classmethod
    def get_size(cls) -> Optional[int]:
        if not os.path.exists(cls._encoder_path) or not os.path.exists(
            cls._decoder_path
        ):
            return None
        return os.stat(cls._encoder_path).st_size + os.stat(cls._decoder_path).st_size

    @classmethod
    def get_modified_at(cls) -> Optional[float]:
        if not os.path.exists(cls._encoder_path) or not os.path.exists(
            cls._decoder_path
        ):
            return None
        return os.stat(cls._encoder_path).st_mtime
