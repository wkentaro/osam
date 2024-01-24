import os

import numpy as np

from samuel._models._base import ModelBase
from samuel._types import ImageEmbedding
from samuel._types import Prompt


class EfficientSam(ModelBase):
    def encode_image(self, image: np.ndarray) -> ImageEmbedding:
        if image.ndim == 2:
            raise ValueError("Grayscale images are not supported")
        if image.ndim == 3 and image.shape[2] == 4:
            raise ValueError("RGBA images are not supported")

        batched_images = image.transpose(2, 0, 1)[None].astype(np.float32) / 255
        image_embedding = self._encoder_session.run(
            output_names=None,
            input_feed={"batched_images": batched_images},
        )[0]

        return ImageEmbedding(
            original_height=image.shape[0],
            original_width=image.shape[1],
            embedding=image_embedding,
        )

    def generate_mask(
        self,
        image_embedding: ImageEmbedding,
        prompt: Prompt,
    ) -> np.ndarray:
        input_point = np.array(prompt.points, dtype=np.float32)
        input_label = np.array(prompt.point_labels, dtype=np.float32)

        # batch_size, num_queries, num_points, 2
        batched_point_coords = input_point[None, None, :, :]
        # batch_size, num_queries, num_points
        batched_point_labels = input_label[None, None, :]

        decoder_inputs = {
            "image_embeddings": image_embedding.embedding,
            "batched_point_coords": batched_point_coords,
            "batched_point_labels": batched_point_labels,
            "orig_im_size": np.array(
                (image_embedding.original_height, image_embedding.original_width),
                dtype=np.int64,
            ),
        }

        masks, _, _ = self._decoder_session.run(None, decoder_inputs)
        mask = masks[0, 0, 0, :, :]  # (1, 1, 3, H, W) -> (H, W)
        mask = mask > 0.0

        return mask


class EfficientSam10m(EfficientSam):
    name = "efficient-sam:10m"

    _encoder_url: str = "https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_encoder.onnx"  # NOQA: E501
    _encoder_md5: str = "2d4a1303ff0e19fe4a8b8ede69c2f5c7"
    _encoder_path: str = os.path.expanduser(
        f"~/.cache/samuel/models/{name}/encoder.onnx"
    )

    _decoder_url: str = "https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_decoder.onnx"  # NOQA: E501
    _decoder_md5: str = "be3575ca4ed9b35821ac30991ab01843"
    _decoder_path: str = os.path.expanduser(
        f"~/.cache/samuel/models/{name}/decoder.onnx"
    )


class EfficientSam25m(EfficientSam):
    name = "efficient-sam:25m"

    _encoder_url: str = "https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx"  # NOQA: E501
    _encoder_md5: str = "7d97d23e8e0847d4475ca7c9f80da96d"
    _encoder_path: str = os.path.expanduser(
        f"~/.cache/samuel/models/{name}/encoder.onnx"
    )

    _decoder_url: str = "https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_decoder.onnx"  # NOQA: E501
    _decoder_md5: str = "d9372f4a7bbb1a01d236b0508300b994"
    _decoder_path: str = os.path.expanduser(
        f"~/.cache/samuel/models/{name}/decoder.onnx"
    )
