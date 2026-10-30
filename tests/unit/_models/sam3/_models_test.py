import types as pytypes

import numpy as np
import pytest

from osam import types
from osam._models.sam3._models import Sam3


@pytest.mark.parametrize(
    "box, expected",
    [
        ([-0.3, -0.3, 0.5, 0.5], (0, 0, 50, 50)),
        ([0.5, 0.5, 1.3, 1.3], (50, 50, 99, 99)),
    ],
)
def test_generate_keeps_mask_of_box_past_image_edge(
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
