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


def test_logits_around_box_keep_the_cells_whose_centers_bracket_the_box() -> None:
    logits = np.tile(np.arange(8, dtype=np.float32) - 4, (8, 1))

    around = types.logits_around_box(
        logits=logits,
        covered_size=(32, 32),
        bounding_box=types.BoundingBox(xmin=9, ymin=0, xmax=14, ymax=31),
    )

    assert around["mask_logits_box"] == (4, 0, 20, 32)
    np.testing.assert_array_equal(
        around["mask_logits"][0], [32000, 32256, 32512, 32768]
    )


def test_logits_around_box_keep_a_cell_for_a_box_thinner_than_one() -> None:
    around = types.logits_around_box(
        logits=np.full((4, 4), 20, np.float32),
        covered_size=(1024, 1024),
        bounding_box=types.BoundingBox(xmin=10, ymin=10, xmax=10, ymax=10),
    )

    np.testing.assert_array_equal(around["mask_logits"], [[32768 + 20 * 256]])
    assert around["mask_logits_box"] == (0, 0, 256, 256)


def test_annotations_from_mask_leave_out_logits_unless_given() -> None:
    (annotation,) = types.annotations_from_mask(mask=np.eye(4, dtype=bool))

    assert annotation.mask_logits is None
    assert annotation.mask_logits_box is None


def _resized(*, logits: np.ndarray, size: tuple[int, int], box=None) -> np.ndarray:
    return np.asarray(
        PIL.Image.fromarray(logits.astype(np.float32)).resize(
            size, resample=PIL.Image.Resampling.BILINEAR, box=box
        )
    )


def _redraw(
    *, annotation: types.Annotation, size: tuple[int, int], scale: int
) -> np.ndarray:
    assert annotation.mask_logits is not None
    assert annotation.mask_logits_box is not None
    xmin, ymin, xmax, ymax = annotation.mask_logits_box
    rows, cols = annotation.mask_logits.shape
    cell_width, cell_height = (xmax - xmin) / cols, (ymax - ymin) / rows
    left, top = max(0, int(xmin)), max(0, int(ymin))
    right, bottom = min(size[0], int(xmax)), min(size[1], int(ymax))
    redrawn = np.zeros((size[1] * scale, size[0] * scale), bool)
    redrawn[top * scale : bottom * scale, left * scale : right * scale] = (
        _resized(
            logits=annotation.mask_logits,
            size=((right - left) * scale, (bottom - top) * scale),
            box=(
                (left - xmin) / cell_width,
                (top - ymin) / cell_height,
                (right - xmin) / cell_width,
                (bottom - ymin) / cell_height,
            ),
        )
        > 32768
    )
    return redrawn


@pytest.mark.parametrize(
    "value, background, width",
    [(10, -1, 1), (-2, 10, 1), (0, 10, 3), (1, -9, 2)],
)
@pytest.mark.parametrize("scale", [1, 4])
def test_mask_logits_redraw_a_thin_column_as_the_model_draws_it(
    value: float, background: float, width: int, scale: int
) -> None:
    logits = np.full((256, 256), background, np.float32)
    logits[:, 128 : 128 + width] = value
    mask = _resized(logits=logits, size=(1024, 1024)) > 0

    (annotation,) = types.annotations_from_mask(
        mask=mask, logits=logits, covered_size=(1024, 1024)
    )

    np.testing.assert_array_equal(
        _redraw(annotation=annotation, size=(1024, 1024), scale=scale),
        _resized(logits=logits, size=(1024 * scale, 1024 * scale)) > 0,
    )
