import numpy as np
import onnx
import onnxruntime
import pytest

from ._models import Sam2Tiny


@pytest.mark.parametrize("pixel_value", [0, 255])
def test_encode_image_normalizes_channels(pixel_value: int) -> None:
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

    expected = (pixel_value / 255 - np.array([0.485, 0.456, 0.406])) / np.array(
        [0.229, 0.224, 0.225]
    )
    np.testing.assert_allclose(
        result.embedding, np.broadcast_to(expected[:, None, None], (3, 8, 8)), rtol=1e-6
    )
