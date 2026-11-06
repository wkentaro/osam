import numpy as np
import pytest

from osam import _types as types
from osam._types import _model


class _VanishingBlob:
    # A blob that osam rm removes right after its first read.
    def __init__(self) -> None:
        self._reads = 0

    def _read(self, *, value: float) -> float | None:
        self._reads += 1
        return value if self._reads == 1 else None

    @property
    def size(self) -> float | None:
        return self._read(value=1)

    @property
    def modified_at(self) -> float | None:
        return self._read(value=2.0)


def test_get_size_reads_each_blob_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_model.Model, "_blobs", {"a": _VanishingBlob()}, raising=False)
    assert _model.Model.get_size() == 1
    monkeypatch.setattr(_model.Model, "_blobs", {"a": _VanishingBlob()}, raising=False)
    assert _model.Model.get_modified_at() == 2.0


class _FakeModel(_model.Model):
    # Decodes to the mask its prompt's single point marks.
    name = "fake"

    def encode_image(self, *, image: np.ndarray) -> types.ImageEmbedding:
        raise NotImplementedError

    @classmethod
    def default_prompt(cls, *, height: int, width: int) -> types.Prompt:
        return types.Prompt(
            points=np.array([[width - 1, height - 1]]), point_labels=np.array([1])
        )

    def _decode(
        self, *, embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> list[types.Annotation]:
        assert prompt.points is not None and prompt.point_labels is not None
        mask = np.zeros((embedding.original_height, embedding.original_width), bool)
        x, y = prompt.points[0].astype(int)
        mask[y, x] = bool(prompt.point_labels[0])
        return types.annotations_from_mask(mask=mask)


def test_decode_uses_the_default_prompt_and_drops_empty_masks() -> None:
    model = _FakeModel.__new__(_FakeModel)
    embedding = types.ImageEmbedding(
        original_height=3, original_width=4, embedding=np.zeros((1, 1, 1), np.float32)
    )

    [annotation] = model.decode(embedding=embedding, prompt=None)
    assert annotation.bounding_box == types.BoundingBox(xmin=3, ymin=2, xmax=3, ymax=2)
    assert annotation.mask is not None and annotation.mask.tolist() == [[True]]

    background = types.Prompt(points=np.zeros((1, 2)), point_labels=np.array([0]))
    assert model.decode(embedding=embedding, prompt=background) == []
