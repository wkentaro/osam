import tracemalloc

import numpy as np
import PIL.Image
import pytest

from osam import _types as types
from osam._models.sam3 import _tracker
from osam._types import _model

from .. import _redraw


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


def _whole_mask(*, logits: np.ndarray, covered_size: tuple[int, int]) -> np.ndarray:
    return types.mask_in_box(
        logits=logits,
        covered_size=covered_size,
        bounding_box=types.BoundingBox(
            xmin=0, ymin=0, xmax=covered_size[0] - 1, ymax=covered_size[1] - 1
        ),
    )


def test_mask_around_its_cells_matches_the_whole_draw_when_shrunk() -> None:
    logits = np.full((288, 288), -0.55859375, np.float32)
    logits[:, 259] = 1.3359375
    covered_size = (2559, 1600)

    annotations = types.annotations_from_logits(
        logits=logits, covered_size=covered_size, height=1600, width=2559
    )

    np.testing.assert_array_equal(
        _full_mask(annotations=annotations, height=1600, width=2559),
        _whole_mask(logits=logits, covered_size=covered_size),
    )


def test_mask_in_a_thin_box_matches_the_whole_draw() -> None:
    logits = np.full((288, 288), -1, np.float32)
    logits[100, 258:260] = [-127.11328125, 22.265625]
    logits[101, 258:260] = [-23.9296875, 124.8671875]
    covered_size = (2557, 287)

    mask = types.mask_in_box(
        logits=logits,
        covered_size=covered_size,
        bounding_box=types.BoundingBox(xmin=2300, ymin=100, xmax=2300, ymax=101),
    )

    np.testing.assert_array_equal(
        mask, _whole_mask(logits=logits, covered_size=covered_size)[100:102, 2300:2301]
    )


def test_mask_in_any_box_matches_the_whole_draw() -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        side = int(rng.choice([256, 288]))
        logits = rng.normal(scale=0.1, size=(side, side)).astype(np.float32)
        covered_size = (int(rng.integers(64, 1200)), int(rng.integers(64, 1200)))
        whole = _whole_mask(logits=logits, covered_size=covered_size)
        for _ in range(10):
            xmin, xmax = np.sort(rng.integers(0, covered_size[0], size=2))
            ymin, ymax = np.sort(rng.integers(0, covered_size[1], size=2))
            match rng.integers(3):
                case 0:
                    xmax = xmin
                case 1:
                    ymax = ymin
            mask = types.mask_in_box(
                logits=logits,
                covered_size=covered_size,
                bounding_box=types.BoundingBox(
                    xmin=int(xmin), ymin=int(ymin), xmax=int(xmax), ymax=int(ymax)
                ),
            )
            np.testing.assert_array_equal(mask, whole[ymin : ymax + 1, xmin : xmax + 1])


def test_mask_in_box_is_row_major() -> None:
    mask = types.mask_in_box(
        logits=np.ones((288, 288), np.float32),
        covered_size=(60, 40),
        bounding_box=types.BoundingBox(xmin=0, ymin=0, xmax=59, ymax=39),
    )

    assert mask.flags.c_contiguous


@pytest.mark.parametrize(
    "width, height, bytes_per_pixel",
    [(16, 20000, 64), (2560, 1600, 4)],
    ids=["tall box narrower than the grid", "whole image"],
)
def test_mask_draw_stays_small(width: int, height: int, bytes_per_pixel: int) -> None:
    logits = np.random.default_rng(0).normal(size=(288, 288)).astype(np.float32)
    covered_size = (width, height)

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    mask = types.mask_in_box(
        logits=logits,
        covered_size=covered_size,
        bounding_box=types.BoundingBox(xmin=0, ymin=0, xmax=width - 1, ymax=height - 1),
    )
    peak = tracemalloc.get_traced_memory()[1] - before
    if not was_tracing:
        tracemalloc.stop()

    np.testing.assert_array_equal(
        mask, _tracker._resize(logits[None, None], height, width)[0, 0] > 0
    )
    assert peak < bytes_per_pixel * mask.size


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


def test_logits_around_box_hold_every_box_on_every_image_size() -> None:
    rng = np.random.default_rng(0)
    logits = np.zeros((288, 288), np.float32)
    for _ in range(2000):
        width, height = (int(side) for side in rng.integers(1, 5000, size=2))
        xmin, xmax = sorted(int(x) for x in rng.integers(0, width, size=2))
        ymin, ymax = sorted(int(y) for y in rng.integers(0, height, size=2))
        if rng.random() < 0.5:
            xmax, ymax = width - 1, height - 1
        bounding_box = types.BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        types.Annotation.model_validate(
            {"bounding_box": bounding_box}
            | types.logits_around_box(
                logits=logits, covered_size=(width, height), bounding_box=bounding_box
            ),
            context={"canvas_size": (width, height)},
        )


def test_annotations_from_logits_leave_out_logits_unless_asked() -> None:
    (annotation,) = types.annotations_from_logits(
        logits=np.eye(4, dtype=np.float32), covered_size=(4, 4), height=4, width=4
    )

    assert annotation.mask_logits is None
    assert annotation.mask_logits_box is None


def test_annotations_from_logits_refuse_cells_their_json_rejects() -> None:
    with pytest.raises(ValueError, match="larger than 4096 pixels a side"):
        types.annotations_from_logits(
            logits=np.ones((1, 4097), np.float32),
            covered_size=(4097, 1),
            height=1,
            width=4097,
            include_mask_logits=True,
        )


@pytest.mark.parametrize("seed", range(20))
def test_mask_logits_redraw_the_mask_on_any_image(seed: int) -> None:
    rng = np.random.default_rng(seed)
    side = int(rng.choice([256, 288]))
    logits = rng.normal(scale=8, size=(side, side)).astype(np.float32) - 6
    width, height = (int(n) for n in rng.integers(1, 3000, size=2))
    longest = max(width, height)
    covered_size = (width, height) if rng.random() < 0.5 else (longest, longest)

    annotations = types.annotations_from_logits(
        logits=logits,
        covered_size=covered_size,
        height=height,
        width=width,
        include_mask_logits=True,
    )
    for _ in range(20):
        xmin, xmax = sorted(int(x) for x in rng.integers(0, width, size=2))
        ymin, ymax = sorted(int(y) for y in rng.integers(0, height, size=2))
        bounding_box = types.BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        annotations.append(
            types.Annotation(
                mask=types.mask_in_box(
                    logits=logits, covered_size=covered_size, bounding_box=bounding_box
                ),
                bounding_box=bounding_box,
                **types.logits_around_box(
                    logits=logits, covered_size=covered_size, bounding_box=bounding_box
                ),
            )
        )

    assert len(annotations) == 21
    for annotation in annotations:
        assert _redraw.distance_where_redraw_differs(annotation=annotation) < 1


def test_mask_logits_redraw_differs_from_the_mask_next_to_a_clipped_cell() -> None:
    logits = np.array([[200, -100], [200, -100]], np.float32)

    (annotation,) = types.annotations_from_logits(
        logits=logits, covered_size=(8, 2), height=2, width=8, include_mask_logits=True
    )

    assert annotation.mask is not None and annotation.mask[:, 4].all()
    assert (_redraw.redrawn(annotation=annotation)[:, 4] < 32768).all()


def _resized(*, logits: np.ndarray, size: tuple[int, int], box=None) -> np.ndarray:
    return np.asarray(
        PIL.Image.fromarray(logits.astype(np.float32)).resize(
            size, resample=PIL.Image.Resampling.BILINEAR, box=box
        )
    )


def _redraw_at(*, annotation: types.Annotation, scale: int) -> np.ndarray:
    assert annotation.mask_logits is not None
    assert annotation.mask_logits_box is not None
    assert annotation.bounding_box is not None
    bb = annotation.bounding_box
    xmin, ymin, xmax, ymax = annotation.mask_logits_box
    rows, cols = annotation.mask_logits.shape
    cell_width, cell_height = (xmax - xmin) / cols, (ymax - ymin) / rows
    box = (
        (bb.xmin - xmin) / cell_width,
        (bb.ymin - ymin) / cell_height,
        (bb.xmax + 1 - xmin) / cell_width,
        (bb.ymax + 1 - ymin) / cell_height,
    )
    size = (scale * (bb.xmax + 1 - bb.xmin), scale * (bb.ymax + 1 - bb.ymin))
    return _resized(logits=annotation.mask_logits, size=size, box=box) > 32768


@pytest.mark.parametrize(
    "value, background, width", [(10, -1, 1), (-2, 10, 1), (0, 10, 3), (1, -9, 2)]
)
@pytest.mark.parametrize("scale", [1, 4])
def test_mask_logits_redraw_a_thin_column_as_the_model_draws_it(
    value: float, background: float, width: int, scale: int
) -> None:
    logits = np.full((256, 256), background, np.float32)
    logits[:, 128 : 128 + width] = value

    (annotation,) = types.annotations_from_logits(
        logits=logits,
        covered_size=(1024, 1024),
        height=1024,
        width=1024,
        include_mask_logits=True,
    )

    assert annotation.bounding_box is not None
    bb = annotation.bounding_box
    direct = _resized(logits=logits, size=(1024 * scale, 1024 * scale)) > 0
    np.testing.assert_array_equal(
        _redraw_at(annotation=annotation, scale=scale),
        direct[
            bb.ymin * scale : (bb.ymax + 1) * scale,
            bb.xmin * scale : (bb.xmax + 1) * scale,
        ],
    )


def test_mask_logits_redraw_a_grid_finer_than_the_image_at_a_larger_size() -> None:
    logits = np.array([[10, -1, 10, -20]], np.float32)

    (annotation,) = types.annotations_from_logits(
        logits=logits, covered_size=(1, 1), height=1, width=1, include_mask_logits=True
    )

    assert annotation.mask_logits is not None
    assert annotation.mask_logits_box is not None
    at_4x = types.Annotation.model_construct(
        bounding_box=types.BoundingBox(xmin=0, ymin=0, xmax=3, ymax=3),
        mask_logits=annotation.mask_logits,
        mask_logits_box=tuple(4 * edge for edge in annotation.mask_logits_box),
    )
    np.testing.assert_array_equal(
        _redraw.redrawn(annotation=at_4x) > 32768,
        np.tile([True, False, True, False], (4, 1)),
    )
