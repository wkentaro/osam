from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import onnxruntime
import pydantic

# An OrtValue stays on the GPU that encoded it; see _onnx.run_on_device.
Feature: TypeAlias = npt.NDArray[np.float32] | onnxruntime.OrtValue


def _nbytes(feature: Feature) -> int:
    if isinstance(feature, np.ndarray):
        return feature.nbytes
    return feature.tensor_size_in_bytes()


class ImageEmbedding(pydantic.BaseModel):
    # An encoded image, kept in memory by Runtime and never sent over HTTP.
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: Feature
    extra_features: list[Feature] = pydantic.Field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return _nbytes(self.embedding) + sum(map(_nbytes, self.extra_features))
