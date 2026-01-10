from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime

from osam import types

from . import _images


def generate_mask_from_image_embedding(
    decoder_session: onnxruntime.InferenceSession,
    image_embedding: types.ImageEmbedding,
    prompt: types.Prompt,
    input_size: int,
) -> npt.NDArray[np.bool_]:
    if prompt.points is None or prompt.point_labels is None:
        raise ValueError("Prompt must contain points and point_labels: %r" % prompt)

    onnx_coord: npt.NDArray[np.float32] = np.concatenate(
        [prompt.points, np.array([[0.0, 0.0]])], axis=0
    )[None, :, :]
    onnx_label: npt.NDArray[np.float32] = np.concatenate(
        [prompt.point_labels, np.array([-1])], axis=0
    )[None, :].astype(np.float32)

    scale: float
    new_height: int
    new_width: int
    scale, new_height, new_width = _images.compute_scale_to_resize_image(
        height=image_embedding.original_height,
        width=image_embedding.original_width,
        target_size=input_size,
    )
    onnx_coord = (
        onnx_coord.astype(float)
        * (
            new_width / image_embedding.original_width,
            new_height / image_embedding.original_height,
        )
    ).astype(np.float32)

    onnx_mask_input: npt.NDArray[np.float32] = np.zeros(
        (1, 1, 256, 256), dtype=np.float32
    )
    onnx_has_mask_input: npt.NDArray[np.float32] = np.array([-1], dtype=np.float32)

    decoder_inputs: dict[str, npt.NDArray] = {
        "image_embeddings": image_embedding.embedding[None, :, :, :],
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(
            (image_embedding.original_height, image_embedding.original_width),
            dtype=np.float32,
        ),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    masks = cast(npt.NDArray[np.bool_], masks)
    mask: npt.NDArray[np.bool_] = masks[0, 0] > 0.0  # (1, 1, H, W) -> (H, W)
    return mask
