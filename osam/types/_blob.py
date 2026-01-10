from __future__ import annotations

import dataclasses
import os
import urllib.parse

import gdown
from loguru import logger


@dataclasses.dataclass
class Blob:
    url: str
    hash: str
    attachments: list[Blob] = dataclasses.field(default_factory=list)

    @property
    def filename(self) -> str:
        return os.path.basename(urllib.parse.urlparse(self.url).path)

    @property
    def path(self) -> str:
        base = os.path.expanduser("~/.cache/osam/models/blobs")
        return f"{base}/{self.hash}"

    @property
    def size(self) -> int | None:
        if not os.path.exists(self.path):
            return None
        total = os.stat(self.path).st_size
        for attachment in self.attachments:
            if not os.path.exists(attachment.path):
                return None
            total += os.stat(attachment.path).st_size
        return total

    @property
    def modified_at(self) -> float | None:
        if not os.path.exists(self.path):
            return None
        latest = os.stat(self.path).st_mtime
        for attachment in self.attachments:
            if not os.path.exists(attachment.path):
                return None
            latest = max(latest, os.stat(attachment.path).st_mtime)
        return latest

    def pull(self):
        gdown.cached_download(url=self.url, path=self.path, hash=self.hash)
        for attachment in self.attachments:
            gdown.cached_download(
                url=attachment.url, path=attachment.path, hash=attachment.hash
            )

    def remove(self):
        if os.path.exists(self.path):
            logger.debug("Removing blob {path!r}", path=self.path)
            os.remove(self.path)
        else:
            logger.warning("Blob {path!r} not found", path=self.path)
        for attachment in self.attachments:
            if os.path.exists(attachment.path):
                logger.debug("Removing attachment {path!r}", path=attachment.path)
                os.remove(attachment.path)
            else:
                logger.warning("Attachment {path!r} not found", path=attachment.path)
