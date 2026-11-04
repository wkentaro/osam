import warnings
from typing import Any
from typing import Final

from ._annotation import Annotation
from ._blob import Blob
from ._blob import PullCancelledError
from ._bounding_box import BoundingBox
from ._model import Device
from ._model_info import ModelInfo
from ._model_info import RunningModel
from ._model_metadata import ModelMetadata

__all__ = [
    "Annotation",
    "Blob",
    "PullCancelledError",
    "BoundingBox",
    "Device",
    "ModelInfo",
    "ModelMetadata",
    "RunningModel",
]


def __getattr__(name: str) -> Any:
    # osam 0.10's request types, kept until osam 1.0 for code written against
    # them.
    DEPRECATED: Final = frozenset(
        {"GenerateRequest", "GenerateResponse", "ImageEmbedding", "Model", "Prompt"}
    )
    if name not in DEPRECATED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    warnings.warn(
        f"osam.types.{name} is deprecated and will be removed in osam 1.0; "
        "use osam.segment",
        DeprecationWarning,
        stacklevel=2,
    )
    from .. import _types  # here, as _types imports this module

    return getattr(_types, name)
