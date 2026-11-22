from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime
import pytest

from osam import _types as types
from osam._models.sam2._models import Sam2Tiny


@pytest.mark.parametrize(
    "pixel_value, expected",
    [
        # torchvision Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # applied to 0 and 1, so the test does not restate the formula under test.
        (0, [-2.1179, -2.0357, -1.8044]),
        (255, [2.2489, 2.4286, 2.6400]),
    ],
)
def test_encode_image_normalizes_channels(
    pixel_value: int, expected: list[float]
) -> None:
    # Pass the encoder input through so normalization is checked without weights.
    graph = onnx.helper.make_graph(
        nodes=[
            onnx.helper.make_node("Identity", inputs=["input"], outputs=[name])
            for name in ["embedding", "high_res1", "high_res2"]
        ],
        name="encoder_input",
        inputs=[
            onnx.helper.make_tensor_value_info(
                "input", onnx.TensorProto.FLOAT, [1, 3, 8, 8]
            )
        ],
        outputs=[
            onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, [1, 3, 8, 8]
            )
            for name in ["embedding", "high_res1", "high_res2"]
        ],
    )
    encoder = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)], ir_version=8
    )
    model = Sam2Tiny.__new__(Sam2Tiny)
    model.input_size = 8
    model._inference_sessions = {
        "encoder": onnxruntime.InferenceSession(
            encoder.SerializeToString(), providers=["CPUExecutionProvider"]
        )
    }

    result = model.encode_image(image=np.full((5, 7, 3), pixel_value, dtype=np.uint8))

    np.testing.assert_allclose(
        result.embedding,
        np.broadcast_to(np.array(expected)[:, None, None], (3, 8, 8)),
        atol=1e-4,
    )


def _clipping_decoder(*, logits: np.ndarray) -> SimpleNamespace:
    def masks(orig_im_size: np.ndarray) -> np.ndarray:
        if tuple(orig_im_size) == logits.shape[-2:]:
            return logits
        return np.zeros((*logits.shape[:2], *orig_im_size), np.float32)

    return SimpleNamespace(
        run=lambda _, feeds: [
            masks(feeds["orig_im_size"]),
            np.array([[1, 0, 0, 0]], np.float32),
            np.clip(logits, -32, 32),
        ]
    )


def test_decode_draws_the_mask_from_unclipped_logits() -> None:
    logits = np.full((1, 4, 256, 256), -5, np.float32)
    logits[..., 0] = 100
    model = Sam2Tiny.__new__(Sam2Tiny)
    model._inference_sessions = {"decoder": _clipping_decoder(logits=logits)}
    embedding = types.ImageEmbedding(
        original_height=1024,
        original_width=1024,
        embedding=np.zeros((1, 1, 1), np.float32),
        extra_features=[np.zeros((1, 1, 1), np.float32)] * 2,
    )

    [annotation] = model.decode(
        embedding=embedding, prompt=types.Prompt(points=[(0, 0)], point_labels=[1])
    )

    assert annotation.bounding_box is not None
    assert annotation.bounding_box.xmax == 5
