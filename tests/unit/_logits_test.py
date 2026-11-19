import numpy as np
import PIL.Image
import pytest

from osam import _logits


def test_mask_from_logits_matches_the_logits_resized_across_row_strips() -> None:
    logits = np.random.default_rng(0).standard_normal((64, 64)).astype(np.float32)
    height, width = 3000, 4096
    x0, y0, x1, y1 = 100, 500, 4096, 2600
    full = np.asarray(
        PIL.Image.fromarray(logits).resize(
            (width, height), resample=PIL.Image.Resampling.BILINEAR
        )
    )

    mask = _logits.mask_from_logits(
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
            _logits.mask_from_logits(
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

    mask = _logits.mask_from_logits(
        logits=logits,
        box=(0, 0, width, height),
        height=height,
        width=width,
        threshold=0,
    )

    np.testing.assert_array_equal(mask, expected)
    assert resizes == [((64, 64), (64, height)), ((64, height), (width, height))]
