from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime

from osam import types


def compute_image_embedding_from_image(
    encoder_session: onnxruntime.InferenceSession, image: npt.NDArray[np.uint8]
) -> types.ImageEmbedding:
    if image.ndim == 2:
        raise ValueError("Grayscale images are not supported")
    if image.ndim == 3 and image.shape[2] == 4:
        raise ValueError("RGBA images are not supported")

    batched_images: npt.NDArray[np.float32] = (
        image.transpose(2, 0, 1)[None].astype(np.float32) / 255
    )
    outputs = encoder_session.run(
        output_names=None,
        input_feed={"batched_images": batched_images},
    )
    image_embedding: npt.NDArray[np.float32] = cast(
        npt.NDArray[np.float32], outputs[0]
    )[0]  # (embedding_dim, height, width)

    return types.ImageEmbedding(
        original_height=image.shape[0],
        original_width=image.shape[1],
        embedding=image_embedding,
    )
