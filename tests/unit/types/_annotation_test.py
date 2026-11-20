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


def test_json_roundtrip_decodes_mask_scores() -> None:
    mask_scores = np.array([[0, 128, 255], [7, 200, 13]], dtype=np.uint8)
    original = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=9, ymax=9),
        mask_scores=mask_scores,
    )

    restored = Annotation.model_validate_json(original.model_dump_json())

    np.testing.assert_array_equal(restored.mask_scores, mask_scores)


@pytest.mark.parametrize(
    "mask_scores", [np.zeros((2, 2), np.float32), np.zeros((2, 2, 3), np.uint8)]
)
def test_rejects_mask_scores_that_are_not_a_uint8_grid(mask_scores) -> None:
    with pytest.raises(pydantic.ValidationError, match="2D uint8"):
        Annotation(
            bounding_box=BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1),
            mask_scores=mask_scores,
        )


def test_rejects_mask_scores_without_a_box() -> None:
    with pytest.raises(pydantic.ValidationError, match="bounding_box is required"):
        Annotation(mask_scores=np.zeros((2, 2), np.uint8))


def _scores_json(*, image: PIL.Image.Image, format: str = "PNG") -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return {
        "bounding_box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1},
        "mask_scores": base64.b64encode(buffer.getvalue()).decode(),
    }


@pytest.mark.parametrize(
    ("image", "format"),
    [
        (PIL.Image.new("RGB", (2, 2)), "PNG"),
        (PIL.Image.new("P", (2, 2)), "PNG"),
        (PIL.Image.fromarray(np.full((2, 2), 300, np.uint16)), "PNG"),
        (PIL.Image.new("L", (2, 2)), "JPEG"),
    ],
)
def test_json_rejects_mask_scores_other_than_a_grayscale_png(image, format) -> None:
    with pytest.raises(pydantic.ValidationError, match="PNG"):
        Annotation.model_validate(_scores_json(image=image, format=format))


@pytest.mark.parametrize(
    ("box", "message"),
    [
        ({"xmin": 1, "ymin": 0, "xmax": 0, "ymax": 1}, "inverted"),
        ({"xmin": -1, "ymin": 0, "xmax": 1, "ymax": 1}, "outside the 2x2 image"),
        ({"xmin": 0, "ymin": 0, "xmax": 2, "ymax": 1}, "outside the 2x2 image"),
    ],
)
def test_json_rejects_mask_scores_on_a_box_off_the_image(box, message) -> None:
    data = _scores_json(image=PIL.Image.new("L", (2, 2))) | {"bounding_box": box}

    with pytest.raises(pydantic.ValidationError, match=message):
        Annotation.model_validate(data, context={"canvas_size": (2, 2)})


def test_rejects_empty_mask_scores() -> None:
    with pytest.raises(pydantic.ValidationError, match="2D uint8"):
        Annotation(
            bounding_box=BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1),
            mask_scores=np.zeros((0, 2), np.uint8),
        )
