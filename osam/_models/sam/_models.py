import abc

import imgviz
import numpy as np
import numpy.typing as npt
from loguru import logger

from osam import types

from . import _decoding
from . import _encoding


class SamBase(types.Model):
    @abc.abstractmethod
    def _generate_mask_from_image_embedding(
        self, image_embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> npt.NDArray[np.bool_]:
        pass

    def generate(self, request: types.GenerateRequest) -> types.GenerateResponse:
        image_embedding: types.ImageEmbedding
        if request.image_embedding is None:
            if request.image is None:
                raise ValueError("request.image or request.image_embedding is required")
            image_embedding = self.encode_image(request.image)
        else:
            image_embedding = request.image_embedding

        prompt: types.Prompt
        if request.prompt is None:
            prompt = types.Prompt(
                points=np.array(
                    [
                        [
                            image_embedding.original_width / 2,
                            image_embedding.original_height / 2,
                        ]
                    ],
                    dtype=np.float32,
                ),
                point_labels=np.array([1], dtype=np.int32),
            )
            logger.warning(
                "Prompt is not given, so using the center point as prompt: {prompt!r}",
                prompt=prompt,
            )
        else:
            prompt = request.prompt
        del request

        if prompt.points is None or prompt.point_labels is None:
            raise ValueError("Prompt must contain points and point_labels: %r", prompt)

        mask: npt.NDArray[np.bool_] = self._generate_mask_from_image_embedding(
            image_embedding=image_embedding, prompt=prompt
        )

        bbox = imgviz.instances.masks_to_bboxes(masks=[mask])[0].astype(int)
        bounding_box: types.BoundingBox = types.BoundingBox(
            ymin=bbox[0], xmin=bbox[1], ymax=bbox[2], xmax=bbox[3]
        )

        return types.GenerateResponse(
            model=self.name,
            image_embedding=image_embedding,
            annotations=[types.Annotation(mask=mask, bounding_box=bounding_box)],
        )


class Sam(SamBase):
    def encode_image(self, image: np.ndarray) -> types.ImageEmbedding:
        return _encoding.compute_image_embedding_from_image(
            encoder_session=self._inference_sessions["encoder"],
            image=image,
        )

    def _generate_mask_from_image_embedding(
        self, image_embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> npt.NDArray[np.bool_]:
        return _decoding.generate_mask_from_image_embedding(
            decoder_session=self._inference_sessions["decoder"],
            image_embedding=image_embedding,
            prompt=prompt,
            input_size=_encoding.get_input_size(
                encoder_session=self._inference_sessions["encoder"]
            ),
        )


class Sam100m(Sam):
    name = "sam:100m"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.encoder.onnx",
            hash="sha256:3346b9cc551c9902fbf3b203935e933592b22e042365f58321c17fc12641fd6a",
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_b_01ec64.quantized.decoder.onnx",
            hash="sha256:edbcf1a0afaa55621fb0abe6b3db1516818b609ea9368f309746a3afc68f7613",
        ),
    }


class Sam300m(Sam):
    name = "sam:300m"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.encoder.onnx",
            hash="sha256:f7158a4a1fe7f670ef47ea2f7f852685425c1ed6caa40f5df86cbe2b0502034f",
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_l_0b3195.quantized.decoder.onnx",
            hash="sha256:552ebb23bf52c5e5b971ac710d1eb8dccfd88b36cc6aff881d1536d1662e6d7b",
        ),
    }


class Sam600m(Sam):
    name = "sam:latest"

    _blobs = {
        "encoder": types.Blob(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.encoder.onnx",
            hash="sha256:a5c745fd4279efc5e5436b412200e983dafc2249ce172af6cc0002a71bb5f485",
        ),
        "decoder": types.Blob(
            url="https://github.com/wkentaro/labelme/releases/download/sam-20230416/sam_vit_h_4b8939.quantized.decoder.onnx",
            hash="sha256:020b385a45ffe51097e1acd10cd791075a86171153505f789a793bc382eef210",
        ),
    }
