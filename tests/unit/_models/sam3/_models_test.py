import types as pytypes

import numpy as np
import pytest

from osam import types
from osam._models.sam3._models import Sam3


@pytest.mark.parametrize(
    "box, expected",
    [
        ([-0.3, -0.3, 0.5, 0.5], (0, 0, 49, 49)),
        ([0.5, 0.5, 1.3, 1.3], (50, 50, 99, 99)),
        ([0.2, 0.3, 0.4, 0.5], (20, 30, 39, 49)),
        ([0.206, 0.306, 0.214, 0.314], (21, 31, 21, 31)),
        ([0.375, 0.125, 0.625, 0.875], (37, 12, 62, 87)),
    ],
)
def test_generate_boxes_the_pixels_inside_the_predicted_edges(
    box: list[float], expected: tuple[int, int, int, int]
) -> None:
    # The decoder predicts box edges, past the image for an object cut off by
    # the image edge, and the box keeps the pixels whose centers they enclose.
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

    response = model.generate(
        types.GenerateRequest(
            model="sam3",
            image_embedding=embedding,
            prompt=types.Prompt(texts=["car"]),
        )
    )

    [annotation] = response.annotations
    bb = annotation.bounding_box
    assert (bb.xmin, bb.ymin, bb.xmax, bb.ymax) == expected
    assert annotation.mask.shape == (bb.ymax - bb.ymin + 1, bb.xmax - bb.xmin + 1)
    assert annotation.mask.all()


@pytest.mark.parametrize(
    "point_labels",
    [[2, 3, 2, 3], [2, 3, 4, 5], [4, 5], [3, 2], [2, 1, 3], [2]],
)
def test_generate_rejects_boxes_without_text_beyond_one_box(
    point_labels: list[int],
) -> None:
    model = Sam3.__new__(Sam3)
    embedding = types.ImageEmbedding(
        original_height=100,
        original_width=100,
        embedding=np.zeros((1, 1, 1), np.float32),
        extra_features=[np.zeros((1, 1, 1), np.float32)] * 3,
    )

    with pytest.raises(ValueError, match=r"texts=\['visual'\]"):
        model.generate(
            types.GenerateRequest(
                model="sam3",
                image_embedding=embedding,
                prompt=types.Prompt(
                    points=np.zeros((len(point_labels), 2)),
                    point_labels=np.array(point_labels),
                ),
            )
        )


@pytest.mark.parametrize(
    "point_labels",
    [[1], [1, 0], [2, 3], [2, 3, 1], [1, 2, 3, 0]],
)
def test_generate_segments_one_object_from_points_and_one_box(
    point_labels: list[int],
) -> None:
    model = Sam3.__new__(Sam3)
    calls: list[list[int]] = []
    model._generate_from_points = lambda **kwargs: calls.append(  # type: ignore[method-assign]
        kwargs["point_labels"].tolist()
    )
    embedding = types.ImageEmbedding(
        original_height=100,
        original_width=100,
        embedding=np.zeros((1, 1, 1), np.float32),
        extra_features=[np.zeros((1, 1, 1), np.float32)] * 3,
    )

    model.generate(
        types.GenerateRequest(
            model="sam3",
            image_embedding=embedding,
            prompt=types.Prompt(
                points=np.zeros((len(point_labels), 2)),
                point_labels=np.array(point_labels),
            ),
        )
    )

    assert calls == [point_labels]
