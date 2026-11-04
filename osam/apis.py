# osam 0.10's Python API, kept until osam 1.0 for code written against it.
# osam.segment and osam.non_maximum_suppression replace it.
import warnings
from typing import Type

import numpy as np
from numpy.typing import NDArray

from . import _apis
from . import _nms
from . import _types as types
from ._apis import registered_model_types  # noqa: F401


def _warn(*, name: str, instead: str) -> None:
    warnings.warn(
        f"osam.apis.{name} is deprecated and will be removed in osam 1.0; "
        f"use {instead}",
        DeprecationWarning,
        stacklevel=3,
    )


def get_model_type_by_name(name: str) -> Type[types.Model]:
    _warn(name="get_model_type_by_name", instead="osam.Client.list")
    return _apis.get_model_type_by_name(name=name)


def get_model_metadata(name: str) -> types.ModelMetadata:
    _warn(name="get_model_metadata", instead="osam.Client.list")
    return _apis.get_model_metadata(name=name)


def generate(request: types.GenerateRequest) -> types.GenerateResponse:
    _warn(name="generate", instead="osam.segment")
    return _apis.generate(request=request)


def non_maximum_suppression(
    boxes: NDArray[np.floating],
    scores: NDArray[np.floating],
    iou_threshold: float,
    score_threshold: float,
    max_num_detections: int,
) -> tuple[
    NDArray[np.floating], NDArray[np.floating], NDArray[np.int64], NDArray[np.int64]
]:
    _warn(name="non_maximum_suppression", instead="osam.non_maximum_suppression")
    return _nms.non_maximum_suppression(
        boxes=boxes,
        scores=scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_num_detections=max_num_detections,
    )
