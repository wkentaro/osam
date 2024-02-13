from typing import Type

from ._base import ModelBase  # noqa: F401
from ._efficient_sam import EfficientSam10m
from ._efficient_sam import EfficientSam25m
from ._sam import Sam91m
from ._sam import Sam308m
from ._sam import Sam636m

MODELS = [
    EfficientSam10m,
    EfficientSam25m,
    Sam91m,
    Sam308m,
    Sam636m,
]


def get_model_class_by_name(name: str) -> Type[ModelBase]:
    model_name: str
    if ":" in name:
        model_name = name
    else:
        model_name = f"{name}:latest"

    for cls in MODELS:
        if cls.name == model_name:
            break
    else:
        raise ValueError(f"Model {name!r} not found.")
    return cls
