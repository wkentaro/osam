import numpy as np
import numpy.typing as npt

from osam import types
from osam._models.sam import SamBase

from . import _decoding
from . import _encoding


class EfficientSam(SamBase):
    def encode_image(self, image: np.ndarray) -> types.ImageEmbedding:
        return _encoding.compute_image_embedding_from_image(
            encoder_session=self._inference_sessions["encoder"],
            image=image,
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
        )


class EfficientSam10m(EfficientSam):
    name = "efficientsam:10m"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_encoder.onnx",
            hash="sha256:7a73ee65aa2c37237c89b4b18e73082f757ffb173899609c5d97a2bbd4ebb02d",
        ),
        "decoder": types.Blob(
            url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vitt_decoder.onnx",
            hash="sha256:e1afe46232c3bfa3470a6a81c7d3181836a94ea89528aff4e0f2d2c611989efd",
        ),
    }


class EfficientSam30m(EfficientSam):
    name = "efficientsam:latest"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_encoder.onnx",
            hash="sha256:4cacbb23c6903b1acf87f1d77ed806b840800c5fcd4ac8f650cbffed474b8896",
        ),
        "decoder": types.Blob(
            url="https://github.com/labelmeai/efficient-sam/releases/download/onnx-models-20231225/efficient_sam_vits_decoder.onnx",
            hash="sha256:4727baf23dacfb51d4c16795b2ac382c403505556d0284e84c6ff3d4e8e36f22",
        ),
    }
