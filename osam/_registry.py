# Every model osam knows, by name.

from . import _errors
from . import _models
from . import _types as types

registered_model_types: list[type[types.Model]] = [
    _models.EfficientSam10m,
    _models.EfficientSam30m,
    _models.Sam100m,
    _models.Sam2BasePlus,
    _models.Sam2Large,
    _models.Sam2Small,
    _models.Sam2Tiny,
    _models.Sam3,
    _models.Sam300m,
    _models.Sam600m,
    _models.YoloWorldXL,
]


def get_model_type_by_name(*, name: str) -> type[types.Model]:
    model_name: str
    if ":" in name:
        model_name = name
    else:
        model_name = f"{name}:latest"

    for cls in registered_model_types:
        if cls.name == model_name:
            break
    else:
        raise _errors.UnknownModelError(f"Model {name!r} not found.")
    return cls


def get_model_metadata(*, name: str) -> types.ModelMetadata:
    return get_model_type_by_name(name=name).metadata
