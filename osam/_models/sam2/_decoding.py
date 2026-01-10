from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime

from osam import types


def generate_mask_from_image_embedding(
    decoder_session: onnxruntime.InferenceSession,
    image_embedding: types.ImageEmbedding,
    prompt: types.Prompt,
    input_size: int,
) -> npt.NDArray[np.bool_]:
    input_point: npt.NDArray[np.float32] = np.array(prompt.points, dtype=np.float32)
    input_point = input_point / np.array(
        [
            image_embedding.original_width / input_size,
            image_embedding.original_height / input_size,
        ],
        dtype=np.float32,
    )
    input_label: npt.NDArray[np.float32] = np.array(
        prompt.point_labels, dtype=np.float32
    )

    decoder_inputs = {
        "image_embeddings": image_embedding.embedding[None],
        "high_res_features1": image_embedding.extra_features[0][None],
        "high_res_features2": image_embedding.extra_features[1][None],
        "point_coords": input_point[None],
        "point_labels": input_label[None],
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.array([0], dtype=np.float32),
        "orig_im_size": np.array(
            (image_embedding.original_height, image_embedding.original_width),
            dtype=np.int64,
        ),
    }
    masks, scores, _low_res_mask = decoder_session.run(None, decoder_inputs)
    masks = cast(npt.NDArray[np.float32], masks)
    scores = cast(npt.NDArray[np.float32], scores)

    mask: npt.NDArray[np.bool_] = (
        masks[0, np.argmax(scores)] > 0.0
    )  # (1, N, H, W) -> (H, W)
    return mask
