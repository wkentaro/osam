from typing import cast

import numpy as np
import numpy.typing as npt
from typing_extensions import override

from osam import _types as types
from osam._models.sam import SamBase


class EfficientSam(SamBase):
    metadata = types.ModelMetadata(
        license_name="Apache License 2.0",
        license_url="https://github.com/wkentaro/efficient-sam/blob/6aebcba09318c4dfe2f9560f7a3f8c42d8b01657/LICENSE",
        source_url="https://github.com/wkentaro/efficient-sam/tree/6aebcba09318c4dfe2f9560f7a3f8c42d8b01657",
        license_spdx="Apache-2.0",
    )

    @override
    def encode_image(self, *, image: npt.NDArray[np.uint8]) -> types.ImageEmbedding:
        batched_images: npt.NDArray[np.float32] = (
            image.transpose(2, 0, 1)[None].astype(np.float32) / 255
        )
        outputs = self._inference_sessions["encoder"].run(
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

    @override
    def _generate_mask_from_image_embedding(
        self,
        *,
        image_embedding: types.ImageEmbedding,
        points: npt.NDArray[np.float32],
        point_labels: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.bool_]:
        # The decoder scales the points by 1024 / size itself.
        scale = 1024 / np.array(
            [image_embedding.original_width, image_embedding.original_height],
            dtype=np.float32,
        )
        points = types.scale_points(points=points, scale=scale) / scale
        decoder_inputs = {
            # batch_size, embedding_dim, height, width
            "image_embeddings": cast(
                npt.NDArray[np.float32], image_embedding.embedding
            )[None],
            # batch_size, num_queries, num_points, 2
            "batched_point_coords": points[None, None, :, :],
            # batch_size, num_queries, num_points
            "batched_point_labels": point_labels[None, None, :],
            "orig_im_size": np.array(
                (image_embedding.original_height, image_embedding.original_width),
                dtype=np.int64,
            ),
        }

        masks, _, _ = self._inference_sessions["decoder"].run(None, decoder_inputs)
        masks = cast(npt.NDArray[np.float32], masks)
        mask: npt.NDArray[np.bool_] = (
            masks[0, 0, 0, :, :] > 0
        )  # (1, 1, 3, H, W) -> (H, W)

        return mask


class EfficientSam10m(EfficientSam):
    name = "efficientsam:10m"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_encoder.onnx",
            hash="sha256:7a73ee65aa2c37237c89b4b18e73082f757ffb173899609c5d97a2bbd4ebb02d",
            download_size=24_799_761,
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_decoder.onnx",
            hash="sha256:e1afe46232c3bfa3470a6a81c7d3181836a94ea89528aff4e0f2d2c611989efd",
            download_size=16_565_728,
        ),
    }


class EfficientSam30m(EfficientSam):
    name = "efficientsam:latest"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx",
            hash="sha256:4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896",
            download_size=89_558_337,
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_decoder.onnx",
            hash="sha256:4727baf23dacfb51d4c16795b2ac382c403505556d0284e84c6ff3d4e8e36f22",
            download_size=16_565_728,
        ),
    }
