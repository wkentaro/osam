from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime

from osam import types

from . import _images


def get_input_size(encoder_session: onnxruntime.InferenceSession) -> int:
    input_height: int
    input_width: int
    input_height, input_width = encoder_session.get_inputs()[0].shape[2:]
    if input_height != input_width:
        raise ValueError("Input height and width must be equal")
    return input_height


def _compute_input_from_image(
    image: npt.NDArray[np.uint8], input_size: int
) -> npt.NDArray[np.float32]:
    _scale, scaled_image = _images.resize_image(image=image, target_size=input_size)
    input_: npt.NDArray[np.float32] = (
        scaled_image.astype(np.float32)
        - np.array([123.675, 116.28, 103.53], dtype=np.float32)
    ) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
    input_ = np.pad(
        input_,
        (
            (0, input_size - input_.shape[0]),
            (0, input_size - input_.shape[1]),
            (0, 0),
        ),
    )
    input_ = input_.transpose(2, 0, 1)[None, :, :, :]
    return input_


def compute_image_embedding_from_image(
    encoder_session: onnxruntime.InferenceSession,
    image: npt.NDArray[np.uint8],
) -> types.ImageEmbedding:
    if image.ndim == 2:
        raise ValueError("Grayscale images are not supported")
    if image.ndim == 3 and image.shape[2] == 4:
        raise ValueError("RGBA images are not supported")

    input_: npt.NDArray[np.float32] = _compute_input_from_image(
        image=image, input_size=get_input_size(encoder_session=encoder_session)
    )
    outputs = encoder_session.run(output_names=None, input_feed={"x": input_})
    image_embedding: npt.NDArray[np.float32] = cast(
        npt.NDArray[np.float32], outputs[0]
    )[0]  # (embedding_dim, height, width)

    return types.ImageEmbedding(
        original_height=image.shape[0],
        original_width=image.shape[1],
        embedding=image_embedding,
    )
