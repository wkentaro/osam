import numpy as np

from osam._models import yoloworld


def test_untransform_bboxes_keeps_boxes_inside_the_image() -> None:
    # A 20x10 image, padded to a 640 square, with a box past its right and
    # bottom edges: bounding boxes are inclusive pixels, so xmax stays below 20.
    bboxes = yoloworld._untransform_bboxes(
        bboxes=np.array([[-64.0, 0.0, 700.0, 700.0]]),
        input_size=640,
        original_image_hw=(10, 20),
        padding_hw=(0, 0),
    )
    assert bboxes.tolist() == [[0, 0, 19, 9]]
