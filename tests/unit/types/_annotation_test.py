import base64
import io
import json
import struct
import zlib

import numpy as np
import PIL.Image
import pydantic
import pytest

from osam.types import Annotation
from osam.types import BoundingBox


def test_json_roundtrip_decodes_mask() -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    original = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=2, ymax=1), mask=mask
    )
    restored = Annotation.model_validate_json(original.model_dump_json())
    np.testing.assert_array_equal(restored.mask, mask)


def test_json_rejects_a_mask_of_another_size_than_its_box_before_decoding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=3, ymax=3),
        mask=np.ones((4, 4), dtype=bool),
    )
    data = json.loads(annotation.model_dump_json())
    data["bounding_box"]["xmax"] = 99
    monkeypatch.setattr(PIL.Image, "open", None)

    with pytest.raises(pydantic.ValidationError, match="bounding box size"):
        Annotation.model_validate(data)


@pytest.mark.parametrize(
    "box",
    [(1, 0, 4, 3), (0, 2, 3, 5), (-1, 0, 2, 3)],
)
def test_json_rejects_a_mask_outside_the_image_it_was_predicted_on(
    box: tuple[int, int, int, int],
) -> None:
    xmin, ymin, xmax, ymax = box
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
        mask=np.ones((4, 4), dtype=bool),
    )
    data = annotation.model_dump_json()

    with pytest.raises(pydantic.ValidationError, match="outside the 4x5 image"):
        Annotation.model_validate_json(data, context={"canvas_size": (4, 5)})


def _png_chunk(*, kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


@pytest.mark.parametrize(
    ("extra_chunk", "message"),
    [
        (
            _png_chunk(
                kind=b"IHDR", data=struct.pack(">IIBBBBB", 64, 64, 1, 0, 0, 0, 0)
            ),
            "one IHDR, not 2",
        ),
        (_png_chunk(kind=b"acTL", data=struct.pack(">II", 1, 0)), "animated"),
        (
            _png_chunk(
                kind=b"fcTL",
                data=struct.pack(">IIIIIHHBB", 0, 2, 2, 2, 2, 0, 0, 0, 0),
            ),
            "animated",
        ),
    ],
    ids=["second-ihdr", "animated", "frame-control"],
)
def test_json_rejects_a_mask_png_pillow_would_read_otherwise_before_decoding_it(
    monkeypatch: pytest.MonkeyPatch, extra_chunk: bytes, message: str
) -> None:
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=3, ymax=3),
        mask=np.ones((4, 4), dtype=bool),
    )
    data = json.loads(annotation.model_dump_json())
    png = base64.b64decode(data["mask"])
    ihdr_end = 8 + 12 + 13
    data["mask"] = base64.b64encode(
        png[:ihdr_end] + extra_chunk + png[ihdr_end:]
    ).decode()
    monkeypatch.setattr(PIL.Image, "open", None)

    with pytest.raises(pydantic.ValidationError, match=message):
        Annotation.model_validate(data)


def test_json_takes_a_mask_with_a_context_of_another_type() -> None:
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=3, ymax=3),
        mask=np.ones((4, 4), dtype=bool),
    )

    restored = Annotation.model_validate_json(
        annotation.model_dump_json(), context=object()
    )

    assert restored.mask is not None and restored.mask.all()


def test_json_roundtrip_decodes_mask_logits() -> None:
    mask_logits = np.array([[0, 32768, 65535], [7, 200, 40000]], dtype=np.uint16)
    original = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=9, ymax=9),
        mask_logits=mask_logits,
        mask_logits_box=(-2.5, 0, 12, 10),
    )

    restored = Annotation.model_validate_json(original.model_dump_json())

    np.testing.assert_array_equal(restored.mask_logits, mask_logits)
    assert restored.mask_logits_box == (-2.5, 0, 12, 10)


@pytest.mark.parametrize(
    "mask_logits",
    [
        np.zeros((2, 2), np.uint8),
        np.zeros((2, 2), np.float32),
        np.zeros((2, 2, 3), np.uint16),
        np.zeros((0, 2), np.uint16),
    ],
)
def test_rejects_mask_logits_that_are_not_a_uint16_grid(mask_logits) -> None:
    with pytest.raises(pydantic.ValidationError, match="2D uint16"):
        Annotation(mask_logits=mask_logits, mask_logits_box=(0, 0, 1, 1))


def test_rejects_mask_logits_without_a_bounding_box_to_draw_in() -> None:
    with pytest.raises(pydantic.ValidationError, match="bounding_box is required"):
        Annotation(
            mask_logits=np.zeros((2, 2), np.uint16), mask_logits_box=(0, 0, 1, 1)
        )


@pytest.mark.parametrize("box", [(1, 0, 4, 3), (0, 2, 3, 5), (-1, 0, 2, 3)])
def test_json_rejects_mask_logits_outside_the_image_they_were_predicted_on(
    box: tuple[int, int, int, int],
) -> None:
    xmin, ymin, xmax, ymax = box
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
        mask_logits=np.zeros((2, 2), np.uint16),
        mask_logits_box=(xmin, ymin, xmax + 1, ymax + 1),
    )
    data = annotation.model_dump_json()

    with pytest.raises(pydantic.ValidationError, match="outside the 4x5 image"):
        Annotation.model_validate_json(data, context={"canvas_size": (4, 5)})


@pytest.mark.parametrize(
    "box", [(20, 0, 30, 10), (1, 0, 10, 10), (0, 1, 10, 10), (0, 0, 9, 10)]
)
def test_json_rejects_a_mask_logits_box_that_misses_a_pixel_of_the_bounding_box(
    box: tuple[float, float, float, float],
) -> None:
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=9, ymax=9),
        mask_logits=np.zeros((2, 2), np.uint16),
        mask_logits_box=(0, 0, 10, 10),
    )
    data = json.loads(annotation.model_dump_json()) | {"mask_logits_box": box}

    with pytest.raises(pydantic.ValidationError, match="misses a pixel center"):
        Annotation.model_validate(data)


@pytest.mark.parametrize("box", [(3, 0, 2, 1), (0, 3, 1, 2)])
def test_rejects_an_inverted_bounding_box(box: tuple[int, int, int, int]) -> None:
    xmin, ymin, xmax, ymax = box
    with pytest.raises(pydantic.ValidationError, match="is inverted"):
        BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def test_accepts_mask_logits_without_a_mask() -> None:
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=0, ymax=0),
        mask_logits=np.zeros((2, 2), np.uint16),
        mask_logits_box=(0, 0, 1, 1),
    )

    assert annotation.mask is None


def test_rejects_mask_logits_without_their_box() -> None:
    with pytest.raises(pydantic.ValidationError, match="come together"):
        Annotation(mask_logits=np.zeros((2, 2), np.uint16))
    with pytest.raises(pydantic.ValidationError, match="come together"):
        Annotation(mask_logits_box=(0, 0, 1, 1))


@pytest.mark.parametrize(
    "box",
    [(1, 0, 0, 1), (0, 1, 1, 1), (0, 0, float("inf"), 1), (float("nan"), 0, 1, 1)],
)
def test_rejects_an_empty_mask_logits_box(box) -> None:
    with pytest.raises(pydantic.ValidationError, match="is empty"):
        Annotation(mask_logits=np.zeros((2, 2), np.uint16), mask_logits_box=box)


def _logits_json(*, image: PIL.Image.Image, format: str = "PNG") -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return {
        "mask_logits": base64.b64encode(buffer.getvalue()).decode(),
        "mask_logits_box": (0, 0, 2, 2),
    }


@pytest.mark.parametrize(
    ("image", "format"),
    [
        (PIL.Image.new("L", (2, 2)), "PNG"),
        (PIL.Image.new("RGB", (2, 2)), "PNG"),
        (PIL.Image.new("P", (2, 2)), "PNG"),
        (PIL.Image.new("L", (2, 2)), "JPEG"),
    ],
)
def test_json_rejects_mask_logits_other_than_a_16_bit_grayscale_png(
    image, format
) -> None:
    with pytest.raises(pydantic.ValidationError, match="PNG"):
        Annotation.model_validate(_logits_json(image=image, format=format))


def test_json_rejects_mask_logits_larger_than_any_grid_before_decoding_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _logits_json(image=PIL.Image.fromarray(np.zeros((1, 4097), np.uint16)))
    monkeypatch.setattr(PIL.Image, "open", None)

    with pytest.raises(pydantic.ValidationError, match="4096 pixels"):
        Annotation.model_validate(data)
