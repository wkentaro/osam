import numpy as np
import numpy.typing as npt

from osam import types
from osam._models.sam import SamBase

from . import _decoding
from . import _encoding


class Sam2(SamBase):
    def encode_image(self, image: npt.NDArray[np.uint8]) -> types.ImageEmbedding:
        return _encoding.compute_image_embedding_from_image(
            encoder_session=self._inference_sessions["encoder"], image=image
        )

    def _generate_mask_from_image_embedding(
        self,
        image_embedding: types.ImageEmbedding,
        prompt: types.Prompt,
    ) -> npt.NDArray[np.bool_]:
        return _decoding.generate_mask_from_image_embedding(
            decoder_session=self._inference_sessions["decoder"],
            image_embedding=image_embedding,
            prompt=prompt,
            input_size=_encoding.get_input_size(
                encoder_session=self._inference_sessions["encoder"]
            ),
        )


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
