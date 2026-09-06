import abc
from typing import cast

import imgviz
import numpy as np
import numpy.typing as npt
import onnxruntime
import PIL.Image
from loguru import logger

from osam import types


class SamBase(types.Model):
    metadata = types.ModelMetadata(
        license_name="Apache License 2.0",
        license_url="https://github.com/wkentaro/segment-anything/blob/13b8480edd05cde9bcf29f9c8a6040b2abe8db56/LICENSE",
        source_url="https://github.com/wkentaro/segment-anything/tree/13b8480edd05cde9bcf29f9c8a6040b2abe8db56",
        license_spdx="Apache-2.0",
    )

    def encode_image(self, image: npt.NDArray[np.uint8]) -> types.ImageEmbedding:
        if image.ndim == 2:
            raise ValueError("Grayscale images are not supported")
        if image.ndim == 3 and image.shape[2] == 4:
            raise ValueError("RGBA images are not supported")
        return self._encode_image(image=image)

    @abc.abstractmethod
    def _encode_image(self, image: npt.NDArray[np.uint8]) -> types.ImageEmbedding:
        pass

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
            raise ValueError(f"Prompt must contain points and point_labels: {prompt!r}")

        mask: npt.NDArray[np.bool_] = self._generate_mask_from_image_embedding(
            image_embedding=image_embedding, prompt=prompt
        )

        bbox = imgviz.masks_to_bboxes(masks=[mask])[0].astype(int)
        bounding_box: types.BoundingBox = types.BoundingBox(
            ymin=bbox[0], xmin=bbox[1], ymax=bbox[2], xmax=bbox[3]
        )
        cropped_mask = mask[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]

        return types.GenerateResponse(
            model=self.name,
            image_embedding=image_embedding,
            annotations=[
                types.Annotation(mask=cropped_mask, bounding_box=bounding_box)
            ],
        )


def get_input_size(encoder_session: onnxruntime.InferenceSession) -> int:
    input_height: int
    input_width: int
    input_height, input_width = encoder_session.get_inputs()[0].shape[2:]
    if input_height != input_width:
        raise ValueError("Input height and width must be equal")
    return input_height


def _compute_resized_hw(height: int, width: int, target_size: int) -> tuple[int, int]:
    if width > height:
        return int(round(height * target_size / width)), target_size
    return target_size, int(round(width * target_size / height))


class Sam(SamBase):
    def _encode_image(self, image: npt.NDArray[np.uint8]) -> types.ImageEmbedding:
        encoder_session = self._inference_sessions["encoder"]
        input_size: int = get_input_size(encoder_session=encoder_session)

        new_height, new_width = _compute_resized_hw(
            height=image.shape[0], width=image.shape[1], target_size=input_size
        )
        scaled_image: npt.NDArray = np.asarray(
            PIL.Image.fromarray(image).resize(
                (new_width, new_height),
                resample=PIL.Image.BILINEAR,  # type: ignore[attr-defined]
            )
        )
        input_: npt.NDArray[np.float32] = (
            scaled_image.astype(np.float32)
            - np.array([123.675, 116.28, 103.53], dtype=np.float32)
        ) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
        input_ = np.pad(
            input_,
            (
                (0, input_size - input_.shape[0]),
                (0, input_size - input_.shape[1]),
                (0, 0),
            ),
        )
        input_ = input_.transpose(2, 0, 1)[None, :, :, :]

        outputs = encoder_session.run(output_names=None, input_feed={"x": input_})
        image_embedding: npt.NDArray[np.float32] = cast(
            npt.NDArray[np.float32], outputs[0]
        )[0]  # (embedding_dim, height, width)

        return types.ImageEmbedding(
            original_height=image.shape[0],
            original_width=image.shape[1],
            embedding=image_embedding,
        )

    def _generate_mask_from_image_embedding(
        self, image_embedding: types.ImageEmbedding, prompt: types.Prompt
    ) -> npt.NDArray[np.bool_]:
        if prompt.points is None or prompt.point_labels is None:
            raise ValueError(f"Prompt must contain points and point_labels: {prompt!r}")

        input_size: int = get_input_size(
            encoder_session=self._inference_sessions["encoder"]
        )

        onnx_coord: npt.NDArray[np.float32] = np.concatenate(
            [prompt.points, np.array([[0.0, 0.0]])], axis=0
        )[None, :, :]
        onnx_label: npt.NDArray[np.float32] = np.concatenate(
            [prompt.point_labels, np.array([-1])], axis=0
        )[None, :].astype(np.float32)

        new_height, new_width = _compute_resized_hw(
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

        decoder_inputs: dict[str, npt.NDArray] = {
            "image_embeddings": image_embedding.embedding[None, :, :, :],
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.array([-1], dtype=np.float32),
            "orig_im_size": np.array(
                (image_embedding.original_height, image_embedding.original_width),
                dtype=np.float32,
            ),
        }

        masks, _, _ = self._inference_sessions["decoder"].run(None, decoder_inputs)
        masks = cast(npt.NDArray[np.bool_], masks)
        mask: npt.NDArray[np.bool_] = masks[0, 0] > 0.0  # (1, 1, H, W) -> (H, W)
        return mask


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
