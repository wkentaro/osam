from typing import cast

import imgviz
import numpy as np
import numpy.typing as npt

from osam import types
from osam._models.sam import SamBase
from osam._models.sam import get_input_size


class Sam2(SamBase):
    metadata = types.ModelMetadata(
        license_name="Apache License 2.0",
        license_url="https://github.com/facebookresearch/sam2/blob/3a7889d905e38ca043c9b5a571ec1635bab678ac/LICENSE",
        source_url="https://github.com/ryouchinsa/sam-cpp-macos/tree/d7399261aab3de35e01172e90da6da4e3191df4e",
        license_spdx="Apache-2.0",
    )

    def _encode_image(self, image: npt.NDArray[np.uint8]) -> types.ImageEmbedding:
        encoder_session = self._inference_sessions["encoder"]

        input_height: int
        input_width: int
        input_height, input_width = encoder_session.get_inputs()[0].shape[2:]

        input_: npt.NDArray[np.float32]
        input_ = (
            imgviz.resize(image, width=input_width, height=input_height).astype(
                np.float32
            )
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

    def _generate_mask_from_image_embedding(
        self,
        image_embedding: types.ImageEmbedding,
        prompt: types.Prompt,
    ) -> npt.NDArray[np.bool_]:
        input_size: int = get_input_size(
            encoder_session=self._inference_sessions["encoder"]
        )

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
        masks, scores, _low_res_mask = self._inference_sessions["decoder"].run(
            None, decoder_inputs
        )
        masks = cast(npt.NDArray[np.float32], masks)
        scores = cast(npt.NDArray[np.float32], scores)

        mask: npt.NDArray[np.bool_] = (
            masks[0, np.argmax(scores)] > 0.0
        )  # (1, N, H, W) -> (H, W)
        return mask


class Sam2Tiny(Sam2):
    name = "sam2:tiny"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_tiny_preprocess.onnx",
            hash="sha256:5557482c56565f6a6c8206874b1a11c392cef8a1766477bf035b919092f2b619",
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_tiny.onnx",
            hash="sha256:11a2c86fabbea9d0268213a9205c99a7f7e379caa0493bd13f5cca8ffaae6777",
        ),
    }


class Sam2Small(Sam2):
    name = "sam2:small"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_small_preprocess.onnx",
            hash="sha256:06016c6dfb146ce10e4dadfdf49e88a05c8d1f97a6b7c57e150e60d2d46a72e7",
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_small.onnx",
            hash="sha256:153aaef5047a3b95285d90cbb39dad6c7b5821bfd944dbf56483f3f735936941",
        ),
    }


class Sam2BasePlus(Sam2):
    name = "sam2:latest"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_base_plus_preprocess.onnx",
            hash="sha256:ce95c44082b4532c25ae01e11da3c9337dab7b04341455c09ae599dc9ae5c438",
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_base_plus.onnx",
            hash="sha256:2ad091af889b20ad2035503b4355cd8924fcf0e29fa6536924c48dc220ecdc56",
        ),
    }


class Sam2Large(Sam2):
    name = "sam2:large"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_large_preprocess.onnx",
            hash="sha256:ab676f957528918496990f242163fd6b41a7222ae255862e846d9ab35115c12e",
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/osam/releases/download/sam2.1/sam2.1_large.onnx",
            hash="sha256:a3ebc6b8e254bd4ca1346901b9472bc2fae9e827cfd67d67e162d0ae2b1ec9a0",
        ),
    }
