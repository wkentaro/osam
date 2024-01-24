import os

import numpy as np
import onnxruntime
import PIL.Image

from samuel._models._base import ModelBase
from samuel._types import ImageEmbedding
from samuel._types import Prompt


class Sam(ModelBase):
    _image_size: int = 1024

    def encode_image(self, image: np.ndarray):
        if image.ndim == 2:
            raise ValueError("Grayscale images are not supported")
        if image.ndim == 3 and image.shape[2] == 4:
            raise ValueError("RGBA images are not supported")

        image_embedding = _compute_image_embedding(
            encoder_session=self._encoder_session,
            image=image,
            image_size=self._image_size,
        )

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
        return _generate_mask(
            decoder_session=self._decoder_session,
            image_embedding=image_embedding,
            prompt=prompt,
            image_size=self._image_size,
        )


def _compute_scale_to_resize_image(height: int, width: int, image_size: int):
    if width > height:
        scale = image_size / width
        new_height = int(round(height * scale))
        new_width = image_size
    else:
        scale = image_size / height
        new_height = image_size
        new_width = int(round(width * scale))
    return scale, new_height, new_width


def _resize_image(image: np.ndarray, image_size: int):
    scale, new_height, new_width = _compute_scale_to_resize_image(
        height=image.shape[0],
        width=image.shape[1],
        image_size=image_size,
    )
    scaled_image = np.asarray(
        PIL.Image.fromarray(image).resize(
            (new_width, new_height), resample=PIL.Image.BILINEAR
        )
    ).astype(np.float32)
    return scale, scaled_image


def _compute_image_embedding(
    encoder_session: onnxruntime.InferenceSession,
    image: np.ndarray,
    image_size: int,
):
    scale, x = _resize_image(image=image, image_size=image_size)
    x = (x - np.array([123.675, 116.28, 103.53], dtype=np.float32)) / np.array(
        [58.395, 57.12, 57.375], dtype=np.float32
    )
    x = np.pad(
        x,
        (
            (0, image_size - x.shape[0]),
            (0, image_size - x.shape[1]),
            (0, 0),
        ),
    )
    x = x.transpose(2, 0, 1)[None, :, :, :]

    output = encoder_session.run(output_names=None, input_feed={"x": x})
    image_embedding = output[0]
    return image_embedding


def _generate_mask(
    decoder_session: onnxruntime.InferenceSession,
    image_embedding: ImageEmbedding,
    prompt: Prompt,
    image_size: int,
):
    onnx_coord = np.concatenate([prompt.points, np.array([[0.0, 0.0]])], axis=0)[
        None, :, :
    ]
    onnx_label = np.concatenate([prompt.point_labels, np.array([-1])], axis=0)[
        None, :
    ].astype(np.float32)

    scale, new_height, new_width = _compute_scale_to_resize_image(
        height=image_embedding.original_height,
        width=image_embedding.original_width,
        image_size=image_size,
    )
    onnx_coord = (
        onnx_coord.astype(float)
        * (
            new_width / image_embedding.original_width,
            new_height / image_embedding.original_height,
        )
    ).astype(np.float32)

    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.array([-1], dtype=np.float32)

    decoder_inputs = {
        "image_embeddings": image_embedding.embedding,
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
    mask = masks[0, 0]  # (1, 1, H, W) -> (H, W)
    mask = mask > 0.0
    return mask


class Sam91m(Sam):
    name = "sam:91m"

    _encoder_url: str = "https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.encoder.onnx"  # NOQA: E501
    _encoder_md5: str = "80fd8d0ab6c6ae8cb7b3bd5f368a752c"
    _encoder_path: str = os.path.expanduser(f"~/.cache/samuel/{name}/encoder.onnx")

    _decoder_url: str = "https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.decoder.onnx"  # NOQA: E501
    _decoder_md5: str = "4253558be238c15fc265a7a876aaec82"
    _decoder_path: str = os.path.expanduser(f"~/.cache/samuel/{name}/decoder.onnx")


class Sam308m(Sam):
    name = "sam:308m"

    _encoder_url: str = "https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.encoder.onnx"  # NOQA: E501
    _encoder_md5: str = "080004dc9992724d360a49399d1ee24b"
    _encoder_path: str = os.path.expanduser(f"~/.cache/samuel/{name}/encoder.onnx")

    _decoder_url: str = "https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.decoder.onnx"  # NOQA: E501
    _decoder_md5: str = "851b7faac91e8e23940ee1294231d5c7"
    _decoder_path: str = os.path.expanduser(f"~/.cache/samuel/{name}/decoder.onnx")


class Sam636m(Sam):
    name = "sam:636m"

    _encoder_url: str = "https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.encoder.onnx"  # NOQA: E501
    _encoder_md5: str = "958b5710d25b198d765fb6b94798f49e"
    _encoder_path: str = os.path.expanduser(f"~/.cache/samuel/{name}/encoder.onnx")

    _decoder_url: str = "https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.decoder.onnx"  # NOQA: E501
    _decoder_md5: str = "a997a408347aa081b17a3ffff9f42a80"
    _decoder_path: str = os.path.expanduser(f"~/.cache/samuel/{name}/decoder.onnx")
