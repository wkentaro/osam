# Every model osam knows, by name.

from collections.abc import Sequence

from . import _models
from . import _types as types


class UnknownModelError(LookupError):
    # A model name osam doesn't know: a 404 from the server.
    pass


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


def get_model_type_by_name(
    *,
    name: str,
    model_types: Sequence[type[types.Model]] = registered_model_types,
) -> type[types.Model]:
    model_name: str
    if ":" in name:
        model_name = name
    else:
        model_name = f"{name}:latest"

    for cls in model_types:
        if cls.name == model_name:
            break
    else:
        raise UnknownModelError(f"Model {name!r} not found.")
    return cls
