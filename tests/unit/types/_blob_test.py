import os
import pathlib
import threading
from collections.abc import Callable
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from unittest import mock

import pytest

from osam.types import _blob
from osam.types._blob import Blob
from osam.types._blob import PullCancelledError
from osam.types._blob import _build_endpoint_url
from osam.types._blob import _resolve_endpoints


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_blob.time, "sleep", lambda seconds: None)


def test_path_standalone_has_no_colon() -> None:
    blob = Blob(
        url="https://example.com/model.onnx",
        hash="sha256:4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896",
    )
    assert os.path.basename(blob.path) == (
        "sha256-4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896"
    )
    assert ":" not in os.path.basename(blob.path)


def test_path_with_attachments_has_no_colon() -> None:
    blob = Blob(
        url="https://example.com/model.onnx",
        hash="sha256:4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896",
        attachments=[Blob(url="https://example.com/config.json", hash="sha256:abc")],
    )
    blob_dir = os.path.basename(os.path.dirname(blob.path))
    assert blob_dir == (
        "sha256-4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896"
    )
    assert os.path.basename(blob.path) == "model.onnx"


def test_resolve_endpoints_unset_is_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSAM_BLOB_ENDPOINT", raising=False)
    assert _resolve_endpoints() == ["direct"]


def test_resolve_endpoints_blank_is_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "  ,  ")
    assert _resolve_endpoints() == ["direct"]


def test_resolve_endpoints_keeps_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", " https://mirror.example.com , direct ")
    assert _resolve_endpoints() == ["https://mirror.example.com", "direct"]


def test_build_endpoint_url_direct_returns_canonical() -> None:
    assert (
        _build_endpoint_url(
            endpoint="direct",
            url="https://example.com/model.onnx",
            hash="sha256:abc123",
        )
        == "https://example.com/model.onnx"
    )


def test_build_endpoint_url_mirror_is_hash_keyed() -> None:
    assert (
        _build_endpoint_url(
            endpoint="https://mirror.example.com/",
            url="https://example.com/model.onnx",
            hash="sha256:abc123",
        )
        == "https://mirror.example.com/abc123"
    )


def test_pull_retries_transient_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OSAM_BLOB_ENDPOINT", raising=False)
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc")

    n_calls = 0

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: object = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        nonlocal n_calls
        n_calls += 1
        if n_calls < 3:
            raise AssertionError("File hash doesn't match")

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ):
        blob.pull()

    assert n_calls == 3


def test_pull_raises_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OSAM_BLOB_ENDPOINT", raising=False)
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc")

    error = AssertionError("File hash doesn't match")
    with mock.patch("osam.types._blob.gdown.cached_download", side_effect=error):
        with pytest.raises(RuntimeError) as excinfo:
            blob.pull()

    assert excinfo.value.__cause__ is error


def test_pull_falls_back_to_direct_when_mirror_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "https://mirror.example.com,direct")
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc123")

    tried: list[str] = []

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: object = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        tried.append(url)
        if url.startswith("https://mirror.example.com"):
            raise RuntimeError("blocked")

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ):
        blob.pull()

    assert tried == [
        "https://mirror.example.com/abc123",
        "https://example.com/model.onnx",
    ]


def test_pull_retries_whole_cycle_not_single_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "https://mirror.example.com,direct")
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc123")

    tried: list[str] = []
    n_direct = 0

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: object = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        nonlocal n_direct
        tried.append(url)
        if url.startswith("https://mirror.example.com"):
            raise RuntimeError("blocked")
        n_direct += 1
        if n_direct < 2:
            raise AssertionError("File hash doesn't match")

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ):
        blob.pull()

    # Each cycle fails over mirror -> direct; the mirror is not retried to
    # exhaustion before direct is tried.
    assert tried == [
        "https://mirror.example.com/abc123",
        "https://example.com/model.onnx",
        "https://mirror.example.com/abc123",
        "https://example.com/model.onnx",
    ]


def test_pull_raises_when_all_endpoints_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "https://mirror.example.com,direct")
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc123")

    error = AssertionError("File hash doesn't match:\nactual: x\nexpected: y")
    with mock.patch("osam.types._blob.gdown.cached_download", side_effect=error):
        with pytest.raises(RuntimeError) as excinfo:
            blob.pull()

    message = str(excinfo.value)
    assert "all endpoints" in message
    assert "[" not in message
    assert "\n" not in message
    assert (
        "https://mirror.example.com/abc123: File hash doesn't match: actual: x "
        "expected: y; https://example.com/model.onnx: File hash doesn't match: "
        "actual: x expected: y" in message
    )
    # 'direct' is already in the list, so the message must not suggest adding it.
    assert "OSAM_BLOB_ENDPOINT" not in message
    # The original failure is preserved as the cause for debugging.
    assert excinfo.value.__cause__ is error


def test_pull_error_suggests_direct_when_mirror_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "https://mirror.example.com")
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc123")

    with mock.patch(
        "osam.types._blob.gdown.cached_download",
        side_effect=RuntimeError("blocked"),
    ):
        with pytest.raises(RuntimeError) as excinfo:
            blob.pull()

    message = str(excinfo.value)
    assert "OSAM_BLOB_ENDPOINT" in message
    assert "direct" in message


def test_pull_uses_mirror_without_contacting_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "https://mirror.example.com,direct")
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc123")

    tried: list[str] = []

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: object = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        tried.append(url)

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ):
        blob.pull()

    assert tried == ["https://mirror.example.com/abc123"]


def test_pull_attachments_use_per_attachment_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # os.path.expanduser uses HOME on POSIX and USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "https://mirror.example.com")
    blob = Blob(
        url="https://example.com/model.onnx",
        hash="sha256:main",
        attachments=[Blob(url="https://example.com/extra.bin", hash="sha256:extra")],
    )

    tried: list[str] = []

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: object = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        tried.append(url)

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ):
        blob.pull()

    assert tried == [
        "https://mirror.example.com/main",
        "https://mirror.example.com/extra",
    ]


def test_pull_quiets_gdown_only_when_progress_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OSAM_BLOB_ENDPOINT", raising=False)
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc")

    quiets: list[bool] = []

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: object = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        quiets.append(quiet)

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ):
        blob.pull()
        blob.pull(progress=lambda filename, bytes_so_far, bytes_total: None)
        blob.pull(cancel=threading.Event())

    assert quiets == [False, True, False]


def test_size_and_modified_at_span_attachments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    blob = Blob(
        url="https://example.com/model.onnx",
        hash="sha256:main",
        attachments=[Blob(url="https://example.com/extra.bin", hash="sha256:extra")],
    )
    os.makedirs(os.path.dirname(blob.path))
    pathlib.Path(blob.path).write_bytes(b"abc")

    assert blob.size is None
    assert blob.modified_at is None

    attachment_path = pathlib.Path(blob.path).with_name("extra.bin")
    attachment_path.write_bytes(b"de")
    os.utime(attachment_path, (2_000_000_000, 2_000_000_000))

    assert blob.size == 5
    assert blob.modified_at == 2_000_000_000


def test_pull_cancel_interrupts_retry_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OSAM_BLOB_ENDPOINT", raising=False)
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc")
    cancel = threading.Event()

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: object = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        cancel.set()
        raise RuntimeError("blocked")

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ) as cached_download:
        with pytest.raises(PullCancelledError):
            blob.pull(cancel=cancel)

    # The backoff returns as soon as the event is set and no retry follows.
    assert cached_download.call_count == 1


def test_pull_cancel_aborts_a_download_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OSAM_BLOB_ENDPOINT", "https://mirror.example.com,direct")
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc123")
    cancel = threading.Event()

    tried: list[str] = []
    reported: list[tuple[str, int, int | None]] = []

    def fake_cached_download(
        url: str,
        path: str,
        hash: str,
        progress: Callable[[int, int | None], None] | None = None,
        quiet: bool = False,
        timeout: object = None,
    ) -> None:
        tried.append(url)
        assert progress is not None
        progress(1024, 4096)
        cancel.set()
        progress(2048, 4096)

    with mock.patch(
        "osam.types._blob.gdown.cached_download", side_effect=fake_cached_download
    ):
        with pytest.raises(PullCancelledError):
            blob.pull(
                progress=lambda filename, bytes_so_far, bytes_total: reported.append(
                    (filename, bytes_so_far, bytes_total)
                ),
                cancel=cancel,
            )

    # Progress still reaches the caller until the cancel, and the cancel is not
    # mistaken for a download failure worth another endpoint.
    assert reported == [("model.onnx", 1024, 4096)]
    assert tried == ["https://mirror.example.com/abc123"]


def test_pull_forwards_timeout_to_gdown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OSAM_BLOB_ENDPOINT", raising=False)
    blob = Blob(url="https://example.com/model.onnx", hash="sha256:abc")

    with mock.patch("osam.types._blob.gdown.cached_download") as cached_download:
        blob.pull()
        blob.pull(timeout=(5, 60))
        blob.pull(timeout=None)

    assert [call.kwargs["timeout"] for call in cached_download.call_args_list] == [
        30,
        (5, 60),
        None,
    ]


@pytest.fixture
def _serve_stalled_download(
    cancel_attempt: int,
) -> Iterator[tuple[str, list[str], threading.Event]]:
    paths: list[str] = []
    cancel = threading.Event()
    stop = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            paths.append(self.path)
            if len(paths) != cancel_attempt:
                self.send_error(503)
                return
            self.send_response(200)
            self.send_header("Content-Length", "1")
            self.end_headers()
            # No body means cancellation must be observed without a progress callback.
            cancel.set()
            stop.wait()

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", paths, cancel
        finally:
            stop.set()
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("cancel_attempt", [1, 3])
def test_pull_cancel_during_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    cancel_attempt: int,
    _serve_stalled_download: tuple[str, list[str], threading.Event],
) -> None:
    url, paths, cancel = _serve_stalled_download
    endpoint = f"{url}/mirror"
    monkeypatch.setenv(
        "OSAM_BLOB_ENDPOINT",
        f"{endpoint},direct" if cancel_attempt == 1 else endpoint,
    )
    monkeypatch.setattr(Blob, "path", property(lambda self: str(tmp_path / "blob")))
    blob = Blob(url=f"{url}/direct", hash="sha256:abc")

    with pytest.raises(PullCancelledError):
        blob.pull(cancel=cancel, timeout=0.1)

    assert paths == ["/mirror/abc"] * cancel_attempt
