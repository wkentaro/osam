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
    def _files(self) -> list[tuple[Blob, str]]:
        blob_dir = os.path.dirname(self.path)
        return [(self, self.path)] + [
            (attachment, os.path.join(blob_dir, attachment.filename))
            for attachment in self.attachments
        ]

    @property
    def size(self) -> int | None:
        total = 0
        for _, path in self._files:
            if not os.path.exists(path):
                return None
            total += os.stat(path).st_size
        return total

    @property
    def modified_at(self) -> float | None:
        latest: float = 0
        for _, path in self._files:
            if not os.path.exists(path):
                return None
            latest = max(latest, os.stat(path).st_mtime)
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

        def _download(blob: Blob, path: str) -> None:
            N_RETRIES: Final = 3
            gdown_progress = _gdown_progress(blob.filename)
            errors: list[str] = []
            last_error: Exception | None = None
            for attempt in range(N_RETRIES):
                errors = []
                for endpoint in endpoints:
                    source = _build_endpoint_url(
                        endpoint=endpoint, url=blob.url, hash=blob.hash
                    )
                    try:
                        gdown.cached_download(
                            url=source,
                            path=path,
                            hash=blob.hash,
                            progress=gdown_progress,
                            quiet=gdown_progress is not None,
                        )
                        return
                    except Exception as e:
                        last_error = e
                        reason = " ".join(str(e).split())
                        logger.warning(
                            "Failed to download {!r} from {!r}: {}",
                            blob.filename,
                            source,
                            reason,
                        )
                        errors.append(f"{source}: {reason}")
                if attempt < N_RETRIES - 1:
                    logger.warning(
                        "Download of {!r} failed on all endpoints "
                        "(attempt {}/{}), retrying in {}s",
                        blob.filename,
                        attempt + 1,
                        N_RETRIES,
                        2**attempt,
                    )
                    time.sleep(2**attempt)
            message = (
                f"Failed to download {blob.filename!r} from all endpoints: "
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

        for blob, path in self._files:
            _download(blob=blob, path=path)

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
