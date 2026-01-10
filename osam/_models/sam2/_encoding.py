from typing import cast

import imgviz
import numpy as np
import numpy.typing as npt
import onnxruntime

from osam import types


def get_input_size(encoder_session: onnxruntime.InferenceSession) -> int:
    input_height: int
    input_width: int
    input_height, input_width = encoder_session.get_inputs()[0].shape[2:]
    if input_height != input_width:
        raise ValueError("Input height and width must be equal")
    return input_height


def compute_image_embedding_from_image(
    encoder_session: onnxruntime.InferenceSession, image: npt.NDArray[np.uint8]
) -> types.ImageEmbedding:
    if image.ndim == 2:
        raise ValueError("Grayscale images are not supported")
    if image.ndim == 3 and image.shape[2] == 4:
        raise ValueError("RGBA images are not supported")

    input_height: int
    input_width: int
    input_height, input_width = encoder_session.get_inputs()[0].shape[2:]

    input_: npt.NDArray[np.float32]
    input_ = (
        imgviz.resize(image, width=input_width, height=input_height).astype(np.float32)
        / 255
    )
    #
    input_ = input_ - np.array([0.485, 0.456, 0.406], dtype=np.float32) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    input_ = input_.transpose(2, 0, 1)[None]

    outputs = encoder_session.run(output_names=None, input_feed={"input": input_})
    image_embedding = cast(npt.NDArray[np.float32], outputs[0])
    high_res_features1 = cast(npt.NDArray[np.float32], outputs[1])
    high_res_features2 = cast(npt.NDArray[np.float32], outputs[2])

    return types.ImageEmbedding(
        original_height=image.shape[0],
        original_width=image.shape[1],
        embedding=image_embedding[0],
        extra_features=[high_res_features1[0], high_res_features2[0]],
    )
