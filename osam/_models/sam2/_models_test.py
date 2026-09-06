import numpy as np
import onnx
import onnxruntime
import pytest

from ._models import Sam2Tiny


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
