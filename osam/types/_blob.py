from __future__ import annotations

import dataclasses
import os
import shutil
import time
import urllib.parse
from collections.abc import Callable
from typing import Final

import gdown
from loguru import logger

_BLOB_ENDPOINT_ENV: Final = "OSAM_BLOB_ENDPOINT"
_DIRECT: Final = "direct"


def _resolve_endpoints() -> list[str]:
    raw = os.environ.get(_BLOB_ENDPOINT_ENV, "")
    endpoints = [entry.strip() for entry in raw.split(",") if entry.strip()]
    return endpoints or [_DIRECT]


def _build_endpoint_url(endpoint: str, url: str, hash: str) -> str:
    if endpoint == _DIRECT:
        return url
    digest = hash.split(":", maxsplit=1)[-1]
    return f"{endpoint.rstrip('/')}/{digest}"


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

        endpoints = _resolve_endpoints()

        def _download(url: str, path: str, hash: str, filename: str) -> None:
            N_RETRIES: Final = 3
            gdown_progress = _gdown_progress(filename)
            errors: list[str] = []
            last_error: Exception | None = None
            for attempt in range(N_RETRIES):
                errors = []
                for endpoint in endpoints:
                    source = _build_endpoint_url(endpoint=endpoint, url=url, hash=hash)
                    try:
                        gdown.cached_download(
                            url=source,
                            path=path,
                            hash=hash,
                            progress=gdown_progress,
                        )
                        return
                    except Exception as e:
                        last_error = e
                        reason = " ".join(str(e).split())
                        logger.warning(
                            "Failed to download {!r} from {!r}: {}",
                            filename,
                            source,
                            reason,
                        )
                        errors.append(f"{source}: {reason}")
                if attempt < N_RETRIES - 1:
                    logger.warning(
                        "Download of {!r} failed on all endpoints "
                        "(attempt {}/{}), retrying in {}s",
                        filename,
                        attempt + 1,
                        N_RETRIES,
                        2**attempt,
                    )
                    time.sleep(2**attempt)
            message = (
                f"Failed to download {filename!r} from all endpoints: "
                f"{'; '.join(errors)}."
            )
            if os.environ.get(_BLOB_ENDPOINT_ENV) and _DIRECT not in endpoints:
                message += (
                    f" Add {_DIRECT!r} to {_BLOB_ENDPOINT_ENV} to fall back to "
                    f"the canonical URL."
                )
            raise RuntimeError(message) from last_error

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
