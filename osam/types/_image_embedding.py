import base64
from typing import Annotated
from typing import Any

import numpy as np
import numpy.typing as npt
import pydantic


def _validate_array(value: Any, info: pydantic.ValidationInfo) -> np.ndarray:
    # Accepts an ndarray, the base64 dict _serialize_array writes, or nested
    # lists, the format embedding was serialized to before.
    name = info.field_name
    arr: np.ndarray
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, dict):
        if {"data", "shape", "dtype"} != set(value.keys()):
            raise ValueError(
                f"{name} must have keys {{'data', 'shape', 'dtype'}}, "
                f"but got {set(value.keys())!r}"
            )
        arr = np.frombuffer(
            base64.b64decode(value["data"]), dtype=value["dtype"]
        ).reshape(value["shape"])
    elif isinstance(value, list):
        arr = np.array(value, dtype=np.float32)
    else:
        raise ValueError(
            f"{name} must be either a numpy array or a dictionary, but got {value!r}"
        )
    if arr.ndim != 3:
        raise ValueError(f"{name} must be of dimension 3, not {arr.ndim!r}")
    if arr.dtype != np.float32:
        raise ValueError(f"{name} must be of type float, but got {arr.dtype!r}")
    return arr


def _serialize_array(arr: np.ndarray) -> dict:
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        # b64encode reads the buffer directly, skipping a tobytes() copy.
        "data": base64.b64encode(np.ascontiguousarray(arr).data).decode(),
    }


_Float32Array = Annotated[
    npt.NDArray[np.float32],
    pydantic.BeforeValidator(_validate_array),
    pydantic.PlainSerializer(_serialize_array),
]


class ImageEmbedding(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    original_height: int
    original_width: int
    embedding: _Float32Array
    extra_features: list[_Float32Array] = pydantic.Field(default_factory=list)
