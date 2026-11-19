import numpy as np
import PIL.Image
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
        self,
        *,
        embedding: types.ImageEmbedding,
        prompt: types.Prompt,
        height: int,
        width: int,
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
    model._decode = lambda *, prompt, **_: prompts.append(prompt) or []
    model.decode(embedding=embedding, prompt=types.Prompt(score_threshold=0.5))
    [prompt] = prompts
    assert prompt.points == [(3, 2)]  # the default prompt, with the thresholds
    assert (prompt.iou_threshold, prompt.score_threshold) == (None, 0.5)


def test_decode_takes_the_prompt_and_gives_annotations_in_original_pixels() -> None:
    model = _FakeModel.__new__(_FakeModel)
    embedding = types.ImageEmbedding(
        original_height=3, original_width=4, embedding=np.zeros((1, 1, 1), np.float32)
    )
    calls: list[tuple[types.Prompt, int, int]] = []
    model._decode = lambda *, prompt, height, width, **_: (
        calls.append((prompt, height, width)) or []
    )

    model.decode(
        embedding=embedding,
        prompt=types.Prompt(points=[(9.5, 5.5)], point_labels=[1]),
        original_size=(16, 9),
    )

    [(prompt, height, width)] = calls
    assert prompt.points == [(2.0, 1.5)]
    assert (height, width) == (9, 16)


@pytest.mark.parametrize(
    "seed, height_range", [(seed, (1000, 1100)) for seed in range(10)] + [(0, (7, 9))]
)
def test_annotations_from_logits_matches_enlarging_the_whole_logits(
    seed: int, height_range: tuple[int, int]
) -> None:
    rng = np.random.default_rng(seed)
    logits = np.full((12, 16), -1.0, np.float32)
    logits[3:9, 4:11] = rng.normal(size=(6, 7)).astype(np.float32)
    height, width = int(rng.integers(*height_range)), int(rng.integers(4300, 4400))

    whole = np.asarray(
        PIL.Image.fromarray(logits).resize(
            (width, height), resample=PIL.Image.Resampling.BILINEAR
        )
    )
    [annotation] = types.annotations_from_logits(
        logits=logits, height=height, width=width, score=0.5
    )

    assert annotation.mask is not None and annotation.bounding_box is not None
    assert annotation.score == 0.5  # noqa: PLR2004
    bb = annotation.bounding_box
    mask = np.zeros((height, width), bool)
    mask[bb.ymin : bb.ymax + 1, bb.xmin : bb.xmax + 1] = annotation.mask
    differs = mask != (whole > 0)
    assert differs.mean() < 1e-4  # noqa: PLR2004
    assert np.abs(whole[differs]).max(initial=0) < 1e-5  # noqa: PLR2004


def test_annotations_from_logits_enlarges_a_diagonal_edge_smoothly() -> None:
    ys, xs = np.mgrid[:16, :16]
    logits = (xs - ys + 0.5).astype(np.float32)

    [annotation] = types.annotations_from_logits(logits=logits, height=192, width=192)

    assert annotation.mask is not None and annotation.bounding_box is not None
    left = annotation.mask.argmax(axis=1) + annotation.bounding_box.xmin
    assert np.abs(np.diff(left[24:-24])).max() <= 2  # noqa: PLR2004


def test_mask_from_logits_matches_the_logits_resized_across_row_strips() -> None:
    logits = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    height, width = 3000, 4096
    x0, y0, x1, y1 = 100, 500, 4096, 2600
    full = np.asarray(
        PIL.Image.fromarray(logits).resize(
            (width, height), resample=PIL.Image.Resampling.BILINEAR
        )
    )

    mask = types.mask_from_logits(
        logits=logits, box=(x0, y0, x1, y1), height=height, width=width, threshold=0
    )

    np.testing.assert_array_equal(mask, full[y0:y1, x0:x1] > 0)


def test_mask_from_logits_matches_the_logits_resized_to_the_image() -> None:
    logits = np.random.default_rng(0).random((36, 36)).astype(np.float32)
    height, width = 100, 160
    full = (
        np.asarray(
            PIL.Image.fromarray(logits).resize(
                (width, height), resample=PIL.Image.Resampling.BILINEAR
            )
        )
        > 0.5
    )
    for x0, y0, x1, y1 in [(7, 3, 62, 51), (0, 0, 160, 100), (150, 90, 160, 100)]:
        np.testing.assert_array_equal(
            types.mask_from_logits(
                logits=logits,
                box=(x0, y0, x1, y1),
                height=height,
                width=width,
                threshold=0.5,
            ),
            full[y0:y1, x0:x1],
        )


def test_mask_from_logits_shrinks_a_shrinking_axis_before_enlarging_the_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    height, width = 8, 5000
    shrunk = PIL.Image.fromarray(logits).resize(
        (64, height), resample=PIL.Image.Resampling.BILINEAR
    )
    expected = (
        np.asarray(
            shrunk.resize((width, height), resample=PIL.Image.Resampling.BILINEAR)
        )
        > 0
    )
    resizes: list[tuple[tuple[int, int], tuple[int, int]]] = []
    resize = PIL.Image.Image.resize

    def recording(self: PIL.Image.Image, size: tuple[int, int], **kwargs):
        resizes.append((self.size, size))
        return resize(self, size, **kwargs)

    monkeypatch.setattr(PIL.Image.Image, "resize", recording)

    mask = types.mask_from_logits(
        logits=logits,
        box=(0, 0, width, height),
        height=height,
        width=width,
        threshold=0,
    )

    np.testing.assert_array_equal(mask, expected)
    assert resizes == [((64, 64), (64, height)), ((64, height), (width, height))]
