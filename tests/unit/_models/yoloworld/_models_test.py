import types

import numpy as np

from osam._models import yoloworld


def test_untransform_bboxes_keeps_boxes_inside_the_image() -> None:
    # A 20x10 image, padded to a 640 square, with a box past its right and
    # bottom edges: bounding boxes are inclusive pixels, so xmax stays below 20.
    bboxes = yoloworld._untransform_bboxes(
        bboxes=np.array([[-64.0, 0.0, 700.0, 700.0]]),
        input_size=640,
        original_image_hw=(10, 20),
    )
    assert bboxes.tolist() == [[0, 0, 19, 9]]


def test_decode_maps_boxes_back_from_the_embedding() -> None:
    # A 10x20 image scales by 32 to 320x640, padded by 160 above and below.
    model = yoloworld.YoloWorldXL.__new__(yoloworld.YoloWorldXL)
    model._inference_sessions = {
        "textual": types.SimpleNamespace(run=lambda *_: [np.ones((2, 4), np.float32)]),
        "yolo": types.SimpleNamespace(
            run=lambda **_: [
                np.array([[[0.9, 0.0]]], np.float32),
                np.array([[[0.0, 160.0, 320.0, 480.0]]], np.float32),
            ]
        ),
    }

    embedding = model.encode_image(image=np.zeros((10, 20, 3), np.uint8))
    assert embedding.embedding.shape == (3, 640, 640)
    [annotation] = model.decode(
        embedding=embedding, prompt=model.default_prompt(height=10, width=20)
    )

    bb = annotation.bounding_box
    assert bb is not None
    assert (bb.xmin, bb.ymin, bb.xmax, bb.ymax) == (0, 0, 10, 9)
    assert annotation.text == "anything"
