import base64
import io
import json
import struct
import zlib

import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import pydantic
import pytest

from osam import _json
from osam.types import Annotation
from osam.types import BoundingBox


def test_json_roundtrip_decodes_mask() -> None:
    mask = np.array([[True, False, True], [False, True, False]])
    original = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=2, ymax=1), mask=mask
    )
    data = original.model_dump_json()
    restored = Annotation.model_validate_json(data)
    np.testing.assert_array_equal(restored.mask, mask)
    png = base64.b64decode(json.loads(data)["mask"])
    assert PIL.Image.open(io.BytesIO(png)).mode == "1"


def test_json_roundtrip_decodes_mask_scores() -> None:
    mask_scores = np.array([[0, 128, 255], [1, 2, 3]], np.uint8)
    restored = Annotation.model_validate_json(
        Annotation(mask_scores=mask_scores).model_dump_json()
    )
    assert restored.mask_scores is not None
    np.testing.assert_array_equal(restored.mask_scores, mask_scores)


def test_json_rejects_mask_scores_of_several_channels() -> None:
    with pytest.raises(pydantic.ValidationError, match="2-D uint8"):
        Annotation.model_validate(
            {
                "mask_scores": _json.image_ndarray_to_b64data(
                    ndarray=np.zeros((2, 2, 3), np.uint8)
                )
            }
        )


def test_json_rejects_a_mask_of_another_size_than_its_box_before_decoding_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=3, ymax=3),
        mask=np.ones((4, 4), dtype=bool),
    )
    data = json.loads(annotation.model_dump_json())
    data["bounding_box"]["xmax"] = 99
    monkeypatch.setattr(PIL.PngImagePlugin, "PngImageFile", None)

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
    monkeypatch.setattr(PIL.PngImagePlugin, "PngImageFile", None)

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


def test_json_decodes_a_mask_past_the_decompression_bomb_limit_in_its_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 10)
    mask = np.eye(8, 9, dtype=bool)
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=1, ymin=2, xmax=9, ymax=9), mask=mask
    )

    decoded = Annotation.model_validate_json(
        annotation.model_dump_json(), context={"canvas_size": (10, 10)}
    )

    assert decoded.mask is not None
    np.testing.assert_array_equal(decoded.mask, mask)


def test_json_rejects_a_large_mask_without_its_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", None)
    monkeypatch.setattr(_json, "MAX_MASK_PIXELS_WITHOUT_IMAGE", 15)
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=3, ymax=3),
        mask=np.ones((4, 4), dtype=bool),
    )
    data = annotation.model_dump_json()

    with pytest.raises(pydantic.ValidationError, match="exceeds 15 pixels"):
        Annotation.model_validate_json(data)
    assert Annotation.model_validate_json(data, context={"canvas_size": (4, 4)})

    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 7)
    monkeypatch.setattr(_json, "MAX_MASK_PIXELS_WITHOUT_IMAGE", 100)
    with pytest.raises(pydantic.ValidationError, match="exceeds 14 pixels"):
        Annotation.model_validate_json(data)


def test_json_rejects_a_mask_of_several_channels_before_loading_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load(self: PIL.Image.Image) -> None:
        raise AssertionError("loaded the pixels")

    png = io.BytesIO()
    PIL.Image.new("RGB", (4, 4)).save(png, format="PNG")
    data = {
        "bounding_box": {"xmin": 0, "ymin": 0, "xmax": 3, "ymax": 3},
        "mask": base64.b64encode(png.getvalue()).decode(),
    }
    monkeypatch.setattr(PIL.PngImagePlugin.PngImageFile, "load", load)

    with pytest.raises(pydantic.ValidationError, match="one channel, not 3"):
        Annotation.model_validate(data)


def test_json_rejects_a_truncated_mask() -> None:
    annotation = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=63, ymax=63),
        mask=np.random.default_rng(0).random((64, 64)) > 0.5,
    )
    data = json.loads(annotation.model_dump_json())
    png = base64.b64decode(data["mask"])
    data["mask"] = base64.b64encode(png[: len(png) // 2]).decode()

    with pytest.raises(pydantic.ValidationError, match="not a PNG file"):
        Annotation.model_validate(data)


def test_json_warns_of_a_mask_past_pillow_s_warning_limit_without_its_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 10)
    data = Annotation(
        bounding_box=BoundingBox(xmin=0, ymin=0, xmax=3, ymax=3),
        mask=np.ones((4, 4), dtype=bool),
    ).model_dump_json()

    with pytest.warns(PIL.Image.DecompressionBombWarning):
        Annotation.model_validate_json(data)
