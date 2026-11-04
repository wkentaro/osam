from typing import Optional
from typing import Type

from . import _models
from . import _types as types

running_model: Optional[types.Model] = None
# The device running_model was asked for; None for automatic selection.
_running_device: Optional[types.Device] = None

registered_model_types: list[Type[types.Model]] = [
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


def get_model_type_by_name(*, name: str) -> Type[types.Model]:
    model_name: str
    if ":" in name:
        model_name = name
    else:
        model_name = f"{name}:latest"

    for cls in registered_model_types:
        if cls.name == model_name:
            break
    else:
        raise ValueError(f"Model {name!r} not found.")
    return cls


def get_model_metadata(*, name: str) -> types.ModelMetadata:
    return get_model_type_by_name(name=name).metadata


def generate(
    *, request: types.GenerateRequest, device: Optional[types.Device] = None
) -> types.GenerateResponse:
    # Runs request.model, reusing the loaded model when model and device match.
    # device: as in Model; None picks one automatically.
    global running_model, _running_device

    model_type = get_model_type_by_name(name=request.model)
    model: types.Model | None = running_model
    if model is None or model.name != model_type.name or _running_device != device:
        # Release the previous model first, so two models are never loaded at once.
        model = running_model = None
        model = running_model = model_type(device=device)
        _running_device = device

    response: types.GenerateResponse = model.generate(request=request)
    return response
