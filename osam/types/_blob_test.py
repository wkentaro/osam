import os

from ._blob import Blob


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
