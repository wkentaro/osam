#!/usr/bin/env python3

import pathlib

import onnx
from onnx import TensorProto
from onnx import helper


def build_model() -> onnx.ModelProto:
    inputs = [
        helper.make_tensor_value_info(
            "boxes", TensorProto.FLOAT, ["num_batches", "num_boxes", 4]
        ),
        helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, ["num_batches", "num_classes", "num_boxes"]
        ),
        helper.make_tensor_value_info(
            "max_output_boxes_per_class", TensorProto.INT64, [1]
        ),
        helper.make_tensor_value_info("iou_threshold", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("score_threshold", TensorProto.FLOAT, [1]),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "selected_indices", TensorProto.INT64, ["num_selected_indices", 3]
        )
    ]
    node = helper.make_node(
        "NonMaxSuppression",
        inputs=[input.name for input in inputs],
        outputs=[output.name for output in outputs],
    )
    graph = helper.make_graph(
        [node], "non_maximum_suppression", inputs=inputs, outputs=outputs
    )
    # NonMaxSuppression last changed in opset 11, and onnx defaults to an IR
    # version newer than onnxruntime accepts; both are pinned low so the graph
    # loads on the widest range of onnxruntime versions.
    model = helper.make_model(
        graph,
        producer_name="osam",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 7
    return model


def main() -> None:
    model = build_model()
    onnx.checker.check_model(model)
    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "osam"
        / "_data"
        / "non_maximum_suppression.onnx"
    )
    onnx.save(model, path)
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
