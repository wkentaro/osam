from __future__ import annotations

import dataclasses
import os
import shutil
import urllib.parse
from collections.abc import Callable

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
        base = os.path.expanduser(
            os.path.join("~", ".cache", "osam", "models", "blobs")
        )
        # Windows can't use ':' in file or directory names
        safe_hash = self.hash.replace("sha256:", "sha256-")
        if self.attachments:
            return os.path.join(base, safe_hash, self.filename)
        else:
            return os.path.join(base, safe_hash)

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

    def pull(
        self,
        progress: Callable[[str, int, int | None], None] | None = None,
    ) -> None:
        def _gdown_progress(
            filename: str,
        ) -> Callable[[int, int | None], None] | None:
            if progress is None:
                return None
            return lambda bytes_so_far, bytes_total: progress(
                filename, bytes_so_far, bytes_total
            )

        def _download(url: str, path: str, hash: str, filename: str) -> None:
            gdown.cached_download(
                url=url,
                path=path,
                hash=hash,
                progress=_gdown_progress(filename),
            )

        if self.attachments:
            blob_dir: str = os.path.dirname(self.path)
            if os.path.isfile(blob_dir):
                logger.warning("Removing file {!r} to create blob directory", blob_dir)
                os.remove(blob_dir)
            os.makedirs(blob_dir, exist_ok=True)
            _download(
                url=self.url,
                path=self.path,
                hash=self.hash,
                filename=self.filename,
            )
            for attachment in self.attachments:
                attachment_path: str = os.path.join(
                    os.path.dirname(self.path), attachment.filename
                )
                _download(
                    url=attachment.url,
                    path=attachment_path,
                    hash=attachment.hash,
                    filename=attachment.filename,
                )
        else:
            _download(
                url=self.url,
                path=self.path,
                hash=self.hash,
                filename=self.filename,
            )

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
