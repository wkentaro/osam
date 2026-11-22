import numpy as np
import pytest

from osam import _types as types
from osam._models.sam3 import _tracker
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


def _full_mask(
    *, annotations: list[types.Annotation], height: int, width: int
) -> np.ndarray:
    full = np.zeros((height, width), bool)
    for annotation in annotations:
        bb = annotation.bounding_box
        assert bb is not None and annotation.mask is not None
        full[bb.ymin : bb.ymax + 1, bb.xmin : bb.xmax + 1] = annotation.mask
    return full


@pytest.mark.parametrize(
    "grid_hw, covered_hw, image_hw",
    [
        ((16, 24), (32, 48), (32, 48)),
        ((16, 16), (40, 40), (25, 40)),
        ((16, 16), (10, 10), (7, 10)),
        ((16, 16), (40, 10), (40, 7)),
    ],
    ids=["covers-the-image", "reaches-past-the-image", "smaller-image", "mixed"],
)
@pytest.mark.parametrize("seed", range(5))
def test_mask_samples_the_logits_at_each_pixel_center(
    grid_hw: tuple[int, int],
    covered_hw: tuple[int, int],
    image_hw: tuple[int, int],
    seed: int,
) -> None:
    logits = np.random.default_rng(seed).normal(size=grid_hw).astype(np.float32) - 1.5
    height, width = image_hw
    expected = (
        _tracker._resize(logits[None, None], *covered_hw)[0, 0, :height, :width] > 0
    )

    annotations = types.annotations_from_logits(
        logits=logits,
        covered_size=(covered_hw[1], covered_hw[0]),
        height=height,
        width=width,
    )

    np.testing.assert_array_equal(
        _full_mask(annotations=annotations, height=height, width=width), expected
    )
    bounding_box = types.BoundingBox(xmin=1, ymin=2, xmax=width - 2, ymax=height - 1)
    np.testing.assert_array_equal(
        types.mask_in_box(
            logits=logits,
            covered_size=(covered_hw[1], covered_hw[0]),
            bounding_box=bounding_box,
        ),
        expected[2:height, 1 : width - 1],
    )


def test_mask_reaches_past_its_positive_cells() -> None:
    logits = np.full((4, 4), -1, np.float32)
    logits[1, 2] = 10
    expected = _tracker._resize(logits[None, None], 32, 32)[0, 0] > 0

    annotations = types.annotations_from_logits(
        logits=logits, covered_size=(32, 32), height=32, width=32
    )

    np.testing.assert_array_equal(
        _full_mask(annotations=annotations, height=32, width=32), expected
    )


def test_mask_is_empty_without_a_positive_cell_over_the_image() -> None:
    logits = np.full((16, 16), -1, np.float32)
    logits[:, -2:] = 1

    assert (
        types.annotations_from_logits(
            logits=logits, covered_size=(64, 64), height=64, width=40
        )
        == []
    )
