from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime

from osam import types


def generate_mask_from_image_embedding(
    decoder_session: onnxruntime.InferenceSession,
    image_embedding: types.ImageEmbedding,
    prompt: types.Prompt,
) -> npt.NDArray[np.bool_]:
    input_point: npt.NDArray[np.float32] = np.array(prompt.points, dtype=np.float32)
    input_label: npt.NDArray[np.float32] = np.array(
        prompt.point_labels, dtype=np.float32
    )

    # batch_size, embedding_dim, height, width
    batched_image_embedding = image_embedding.embedding[None, :, :, :]
    # batch_size, num_queries, num_points, 2
    batched_point_coords = input_point[None, None, :, :]
    # batch_size, num_queries, num_points
    batched_point_labels = input_label[None, None, :]

    decoder_inputs = {
        "image_embeddings": batched_image_embedding,
        "batched_point_coords": batched_point_coords,
        "batched_point_labels": batched_point_labels,
        "orig_im_size": np.array(
            (image_embedding.original_height, image_embedding.original_width),
            dtype=np.int64,
        ),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    masks = cast(npt.NDArray[np.bool_], masks)
    mask: npt.NDArray[np.bool_] = masks[0, 0, 0, :, :] > 0  # (1, 1, 3, H, W) -> (H, W)

    return mask
