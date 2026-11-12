import types as pytypes

import numpy as np
import PIL.Image
import pytest

from osam import _types as types
from osam._models.sam3 import _models
from osam._models.sam3._models import Sam3


@pytest.mark.parametrize(
    "box, expected",
    [
        ([-0.3, -0.3, 0.5, 0.5], (0, 0, 50, 50)),
        ([0.5, 0.5, 1.3, 1.3], (50, 50, 99, 99)),
    ],
)
def test_decode_keeps_mask_of_box_past_image_edge(
    box: list[float], expected: tuple[int, int, int, int]
) -> None:
    # The decoder can predict a box past the image for an object cut off by
    # the edge.
    language_encoder = pytypes.SimpleNamespace(
        run=lambda *_: [np.ones((1, 32), bool), np.zeros((32, 1, 8), np.float32)]
    )
    decoder = pytypes.SimpleNamespace(
        get_inputs=lambda: [pytypes.SimpleNamespace(name="box_coords", shape=[1])],
        run=lambda *_: [
            np.array([box], np.float32),
            np.array([0.9], np.float32),
            np.ones((1, 1, 8, 8), np.float32),
        ],
    )
    model = Sam3.__new__(Sam3)
    model._inference_sessions = {
        "language_encoder": language_encoder,
        "decoder": decoder,
    }
    embedding = types.ImageEmbedding(
        original_height=100,
        original_width=100,
        embedding=np.zeros((1, 1, 1), np.float32),
        extra_features=[np.zeros((1, 1, 1), np.float32)] * 3,
    )

    annotations = model.decode(
        embedding=embedding,
        prompt=types.Prompt(texts=["car"]),
    )

    [annotation] = annotations
    bb = annotation.bounding_box
    assert (bb.xmin, bb.ymin, bb.xmax, bb.ymax) == expected
    assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)
    assert annotation.mask.all()


@pytest.mark.parametrize(
    "point_labels, box, expected",
    [
        ([1], None, [1]),
        ([1, 0], None, [1, 0]),
        (None, (0, 0, 1, 1), [2, 3]),
        ([1, 0], (0, 0, 1, 1), [1, 0, 2, 3]),
    ],
)
def test_decode_segments_one_object_from_points_and_one_box(
    point_labels: list[int] | None,
    box: tuple[float, float, float, float] | None,
    expected: list[int],
) -> None:
    model = Sam3.__new__(Sam3)
    calls: list[list[int]] = []
    model._generate_from_points = lambda **kwargs: calls.append(
        kwargs["point_labels"].tolist()
    )
    embedding = types.ImageEmbedding(
        original_height=100,
        original_width=100,
        embedding=np.zeros((1, 1, 1), np.float32),
        extra_features=[np.zeros((1, 1, 1), np.float32)] * 3,
    )

    model.decode(
        embedding=embedding,
        prompt=types.Prompt.model_validate(
            {
                "points": None
                if point_labels is None
                else [[0, 0]] * len(point_labels),
                "point_labels": point_labels,
                "box": box,
            }
        ),
    )

    assert calls == [expected]


def test_decode_finds_objects_for_each_text() -> None:
    language_encoder = pytypes.SimpleNamespace(
        run=lambda *_: [
            np.ones((1, 32), bool),
            np.zeros((32, 1, 8), np.float32),
        ]
    )
    # The first run, for "car", finds two objects; the second, for "dog", one.
    runs = iter([[0.9, 0.2], [0.5]])

    def run(*_):
        scores = np.array(next(runs), np.float32)
        return [
            np.tile(np.array([[0.1, 0.1, 0.5, 0.5]], np.float32), (len(scores), 1)),
            scores,
            np.ones((len(scores), 1, 8, 8), np.float32),
        ]

    decoder = pytypes.SimpleNamespace(
        get_inputs=lambda: [pytypes.SimpleNamespace(name="box_coords", shape=[1])],
        run=run,
    )
    model = Sam3.__new__(Sam3)
    model._inference_sessions = {
        "language_encoder": language_encoder,
        "decoder": decoder,
    }
    embedding = types.ImageEmbedding(
        original_height=100,
        original_width=100,
        embedding=np.zeros((1, 1, 1), np.float32),
        extra_features=[np.zeros((1, 1, 1), np.float32)] * 3,
    )

    annotations = model.decode(
        embedding=embedding,
        prompt=types.Prompt(texts=["car", "dog"], max_annotations=2),
    )

    # The cap applies across texts, keeping the highest scores.
    assert [(a.text, a.score) for a in annotations] == [
        ("car", pytest.approx(0.9)),
        ("dog", pytest.approx(0.5)),
    ]


def test_encode_image_fetches_only_the_outputs_decoders_read() -> None:
    requested: list[list[str]] = []

    def run(output_names: list[str], inputs: dict) -> list[np.ndarray]:
        requested.append(output_names)
        return [
            np.full((1, 1, 1, 1), index, np.float32)
            for index in range(len(output_names))
        ]

    model = Sam3.__new__(Sam3)
    model._inference_sessions = {
        "image_encoder": pytypes.SimpleNamespace(
            run=run, get_providers=lambda: ["CPUExecutionProvider"]
        )
    }

    embedding = model.encode_image(image=np.zeros((4, 6, 3), np.uint8))

    [output_names] = requested
    assert "vision_pos_enc_0" not in output_names
    assert "vision_pos_enc_1" not in output_names
    # Each output lands in the field its decoder reads it from.
    by_name = dict(zip(output_names, range(len(output_names))))
    assert embedding.embedding.item() == by_name["backbone_fpn_0"]
    assert [feature.item() for feature in embedding.extra_features] == [
        by_name[name]
        for name in [
            "backbone_fpn_1",
            "backbone_fpn_2",
            "vision_pos_enc_2",
            "image_embeddings",
            "high_res_features1",
            "high_res_features2",
        ]
    ]


@pytest.mark.parametrize(
    "device, ort_version, expected",
    [
        ("cpu", (1, 30), "image_encoder"),
        ("cuda", (1, 30), "image_encoder_fused_fp16"),
        ("webgpu", (1, 30), "image_encoder_fused_fp16"),
        ("coreml", (1, 30), "image_encoder_fp16"),
        ("coreml", (1, 23), "image_encoder"),
    ],
)
def test_file_runs_fastest_on_the_device(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    ort_version: tuple[int, int],
    expected: str,
) -> None:
    from osam._models.sam3 import _models

    monkeypatch.setattr(_models, "_ORT_VERSION", ort_version)
    assert _models._file(name="image_encoder", device=device) == expected
    assert _models._file(name="decoder", device="cpu") == "decoder"


def test_remove_deletes_the_files_an_upgrade_leaves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osam import types
    from osam._models.sam3._models import Sam3

    removed: list[str] = []
    monkeypatch.setattr(types.Blob, "remove", lambda self: removed.append(self.hash))
    Sam3.remove()
    # The v0.6.0 image encoder, which v0.7.0 replaced.
    assert (
        "sha256:c6b984d95ca7ca0564356253b92ee30287188cfffab5d4f87709b79d5d9aa6c8"
        in removed
    )


def test_crop_mask_matches_the_mask_resized_to_the_image() -> None:
    # As sam3-onnx resizes the whole mask, on boxes that end on odd native
    # pixels and on the image's edges.
    mask = np.random.default_rng(0).random((36, 36)).astype(np.float32)
    height, width = 100, 160
    full = (
        np.asarray(
            PIL.Image.fromarray(mask).resize(
                (width, height), resample=PIL.Image.Resampling.BILINEAR
            )
        )
        > 0.5
    )
    for xmin, ymin, xmax, ymax in [(7, 3, 61, 50), (0, 0, 159, 99), (150, 90, 159, 99)]:
        bounding_box = types.BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        np.testing.assert_array_equal(
            _models._crop_mask(
                mask=mask, bounding_box=bounding_box, height=height, width=width
            ),
            full[ymin : ymax + 1, xmin : xmax + 1],
        )
