from __future__ import annotations

import dataclasses
import os
import shutil
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
        if self.attachments:
            # Windows can't use ':' for directory names
            safe_hash = self.hash.replace("sha256:", "sha256-")
            return f"{base}/{safe_hash}/{self.filename}"
        else:
            return f"{base}/{self.hash}"

    @property
    def size(self) -> int | None:
        if not os.path.exists(self.path):
            return None
        total = os.stat(self.path).st_size
        for attachment in self.attachments:
            attachment_path: str = os.path.join(
                os.path.dirname(self.path), attachment.filename
            )
            if not os.path.exists(attachment_path):
                return None
            total += os.stat(attachment_path).st_size
        return total

    @property
    def modified_at(self) -> float | None:
        if not os.path.exists(self.path):
            return None
        latest = os.stat(self.path).st_mtime
        for attachment in self.attachments:
            attachment_path: str = os.path.join(
                os.path.dirname(self.path), attachment.filename
            )
            if not os.path.exists(attachment_path):
                return None
            latest = max(latest, os.stat(attachment_path).st_mtime)
        return latest

    def pull(self):
        if self.attachments:
            blob_dir: str = os.path.dirname(self.path)
            if os.path.isfile(blob_dir):
                logger.warning("Removing file {!r} to create blob directory", blob_dir)
                os.remove(blob_dir)
            os.makedirs(blob_dir, exist_ok=True)
            gdown.cached_download(url=self.url, path=self.path, hash=self.hash)
            for attachment in self.attachments:
                attachment_path: str = os.path.join(
                    os.path.dirname(self.path), attachment.filename
                )
                gdown.cached_download(
                    url=attachment.url, path=attachment_path, hash=attachment.hash
                )
        else:
            gdown.cached_download(url=self.url, path=self.path, hash=self.hash)

    def remove(self):
        if self.attachments:
            dir_path = os.path.dirname(self.path)
            if os.path.exists(dir_path):
                logger.debug("Removing blob directory {!r}", dir_path)
                shutil.rmtree(dir_path)
            else:
                logger.warning("Blob directory {!r} not found", dir_path)
        else:
            if os.path.exists(self.path):
                logger.debug("Removing blob {!r}", self.path)
                os.remove(self.path)
            else:
                logger.warning("Blob {!r} not found", self.path)
