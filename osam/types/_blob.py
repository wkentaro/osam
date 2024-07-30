import dataclasses
import os

import gdown
from loguru import logger


@dataclasses.dataclass
class Blob:
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
            logger.debug("Removing blob {path!r}", path=self.path)
            os.remove(self.path)
        else:
            logger.warning("Blob {path!r} not found", path=self.path)
