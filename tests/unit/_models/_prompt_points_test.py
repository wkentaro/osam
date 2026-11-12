import types as pytypes

import numpy as np
import pytest

from osam import types
from osam._models.efficientsam._models import EfficientSam10m
from osam._models.sam._models import Sam100m
from osam._models.sam2._models import Sam2Tiny
from osam._models.sam3._models import Sam3

HEIGHT, WIDTH = 128, 256


def _session(*, inputs: list, outputs: list) -> pytypes.SimpleNamespace:
    return pytypes.SimpleNamespace(
        get_inputs=lambda: [pytypes.SimpleNamespace(shape=[1, 3, 1024, 1024])],
        run=lambda _, feeds: inputs.append(feeds) or outputs,
    )


@pytest.mark.parametrize(
    "model_type, decoder, coords, outputs, model_size",
    [
        # Each decoder adds 0.5 to read an integer as a pixel center in the
        # image it was encoded at; EfficientSAM resizes the points first.
        (
            Sam100m,
            "decoder",
            "point_coords",
            [np.ones((1, 1, HEIGHT, WIDTH))] * 3,
            (1024, 512),
        ),
        (
            Sam2Tiny,
            "decoder",
            "point_coords",
            [np.ones((1, 1, HEIGHT, WIDTH)), np.ones((1, 1)), None],
            (1024, 1024),
        ),
        (
            EfficientSam10m,
            "decoder",
            "batched_point_coords",
            [np.ones((1, 1, 3, HEIGHT, WIDTH))] * 3,
            (1024, 1024),
        ),
        (
            Sam3,
            "point_decoder",
            "point_coords",
            [np.ones((1, 1)), np.zeros((1, 1, 8, 8), np.float32)],
            (1008, 1008),
        ),
    ],
)
def test_generate_sends_the_clicked_pixel_center_at_its_place_in_the_model(
    model_type: type[types.Model],
    decoder: str,
    coords: str,
    outputs: list,
    model_size: tuple[int, int],
) -> None:
    inputs: list[dict] = []
    model = model_type.__new__(model_type)
    model._inference_sessions = {
        "encoder": _session(inputs=[], outputs=[]),
        decoder: _session(inputs=inputs, outputs=outputs),
    }
    embedding = types.ImageEmbedding(
        original_height=HEIGHT,
        original_width=WIDTH,
        embedding=np.zeros((1, 1, 1), np.float32),
        extra_features=[np.zeros((1, 1, 1), np.float32)] * 6,
    )

    model.generate(
        types.GenerateRequest(
            model=model.name,
            image_embedding=embedding,
            prompt=types.Prompt(
                points=np.array([(10, 20), (WIDTH - 1, 0)]),
                point_labels=np.array([1, 1]),
            ),
        )
    )

    sent = inputs[0][coords].reshape(-1, 2)[:2]
    assert sent.dtype == np.float32
    if model_type is EfficientSam10m:
        sent = sent * [1024 / WIDTH, 1024 / HEIGHT]
    scale = np.array(model_size) / [WIDTH, HEIGHT]
    np.testing.assert_allclose(
        sent + 0.5, (np.array([(10, 20), (WIDTH - 1, 0)]) + 0.5) * scale, rtol=1e-6
    )
