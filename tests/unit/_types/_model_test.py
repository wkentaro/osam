import numpy as np
import pytest

from osam import _types as types
from osam._types import _model


class _VanishingBlob:
    # A blob that osam rm removes right after its first read.

    _reads: int

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
    prompt_kinds = frozenset({"points"})

    def encode_image(self, *, image: np.ndarray) -> types.ImageEmbedding:
        raise NotImplementedError

    @classmethod
    def default_prompt(cls, *, height: int, width: int) -> types.Prompt:
        return types.Prompt(points=[(width - 1, height - 1)], point_labels=[1])

    def _decode(
        self, *, embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> list[types.Annotation]:
        assert prompt.points is not None and prompt.point_labels is not None
        mask = np.zeros((embedding.original_height, embedding.original_width), bool)
        x, y = map(int, prompt.points[0])
        mask[y, x] = bool(prompt.point_labels[0])
        return types.annotations_from_mask(mask=mask)


def test_decode_uses_the_default_prompt_and_drops_empty_masks() -> None:
    model = _FakeModel.__new__(_FakeModel)
    embedding = types.ImageEmbedding(
        original_height=3, original_width=4, embedding=np.zeros((1, 1, 1), np.float32)
    )

    [annotation] = model.decode(embedding=embedding, prompt=types.Prompt())
    assert annotation.bounding_box == types.BoundingBox(xmin=3, ymin=2, xmax=3, ymax=2)
    assert annotation.mask is not None and annotation.mask.tolist() == [[True]]

    background = types.Prompt(points=[(0, 0)], point_labels=[0])
    assert model.decode(embedding=embedding, prompt=background) == []


def test_decode_rejects_other_kinds_and_keeps_thresholds() -> None:
    model = _FakeModel.__new__(_FakeModel)
    embedding = types.ImageEmbedding(
        original_height=3, original_width=4, embedding=np.zeros((1, 1, 1), np.float32)
    )
    with pytest.raises(ValueError, match="fake takes points, not texts"):
        model.decode(embedding=embedding, prompt=types.Prompt(texts=["dog"]))

    prompts: list[types.Prompt] = []
    model._decode = lambda *, embedding, prompt: prompts.append(prompt) or []
    model.decode(embedding=embedding, prompt=types.Prompt(score_threshold=0.5))
    [prompt] = prompts
    assert prompt.points == [(3, 2)]  # the default prompt, with the thresholds
    assert (prompt.iou_threshold, prompt.score_threshold) == (None, 0.5)


def test_scores_in_box_tile_the_box_at_twice_the_grid_resolution() -> None:
    probabilities = np.tile(np.array([0, 1, 2, 3], np.float32) / 3, (4, 1))

    scores = types.scores_in_box(
        probabilities=probabilities,
        covered_size=(16, 16),
        bounding_box=types.BoundingBox(xmin=4, ymin=0, xmax=11, ymax=15),
    )

    assert scores.shape == (8, 4)
    np.testing.assert_array_equal(scores[0], [64, 106, 149, 191])


def test_scores_in_box_keep_a_cell_for_a_box_thinner_than_one() -> None:
    scores = types.scores_in_box(
        probabilities=np.ones((4, 4), np.float32),
        covered_size=(1024, 1024),
        bounding_box=types.BoundingBox(xmin=10, ymin=10, xmax=10, ymax=10),
    )

    np.testing.assert_array_equal(scores, [[255]])


def test_annotations_from_mask_leave_out_scores_without_probabilities() -> None:
    (annotation,) = types.annotations_from_mask(mask=np.eye(4, dtype=bool))

    assert annotation.mask_scores is None


def test_scores_in_box_size_follows_the_box_not_its_position() -> None:
    probabilities = np.zeros((288, 288), np.float32)

    shapes = {
        types.scores_in_box(
            probabilities=probabilities,
            covered_size=(1920, 1080),
            bounding_box=types.BoundingBox(xmin=xmin, ymin=0, xmax=xmin + 99, ymax=19),
        ).shape
        for xmin in (21, 341)
    }

    assert len(shapes) == 1
