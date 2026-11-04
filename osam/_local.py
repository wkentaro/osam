# Runs models in this process, for osam.segment and osam.apis.
import threading
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from . import _json
from . import _registry
from . import _requests
from . import _types as types

running_model: Optional[types.Model] = None
# The device running_model was asked for; None for automatic selection.
_running_device: Optional[types.Device] = None
# The last image's embedding, by model name and image hash, so several prompts
# on one image encode it once.
# ponytail: one image, keep more if callers alternate between images.
_last_embedding: Optional[tuple[tuple[str, str], types.ImageEmbedding]] = None
# Calls from several threads take turns on the one running model.
_lock = threading.Lock()


def generate(
    *, request: types.GenerateRequest, device: Optional[types.Device] = None
) -> types.GenerateResponse:
    # Runs request.model, reusing the loaded model when model and device match.
    # device: as in Model; None picks one automatically.
    with _lock:
        model: types.Model = _run(name=request.model, device=device)
        return model.generate(request=request)


def segment(
    model: str,
    image: np.ndarray,
    *,
    points: Optional[ArrayLike] = None,
    point_labels: Optional[ArrayLike] = None,
    box: Optional[ArrayLike] = None,
    texts: Optional[list[str]] = None,
    example_boxes: Optional[ArrayLike] = None,
    example_box_labels: Optional[ArrayLike] = None,
    iou_threshold: Optional[float] = None,
    score_threshold: Optional[float] = None,
    max_annotations: Optional[int] = None,
    device: Optional[types.Device] = None,
) -> list[types.Annotation]:
    """Segment image, an HxWx3 uint8 array, with model, in this process.

    Takes the prompt as osam.Client.segment does, and raises ValueError for one
    the model doesn't take. The model stays loaded until a call asks for
    another model or device, and further prompts on the same image skip its
    encoder. device: "cpu" or "cuda"; None picks one automatically.
    """
    global _last_embedding

    request = _requests.SegmentRequest(
        model=model,
        image=image,
        **_requests.to_json_fields(
            fields={
                "points": points,
                "point_labels": point_labels,
                "box": box,
                "texts": texts,
                "example_boxes": example_boxes,
                "example_box_labels": example_box_labels,
                "iou_threshold": iou_threshold,
                "score_threshold": score_threshold,
                "max_annotations": max_annotations,
            }
        ),
    )
    height, width = image.shape[:2]
    prompt: Optional[types.Prompt] = request.to_prompt(
        model_type=_registry.get_model_type_by_name(name=model),
        height=height,
        width=width,
    )
    with _lock:
        running: types.Model = _run(name=model, device=device)
        key: tuple[str, str] = (running.name, _json.hash_image(image=image))
        cached: Optional[types.ImageEmbedding] = (
            _last_embedding[1]
            if _last_embedding is not None and _last_embedding[0] == key
            else None
        )
        # Models use the embedding when given one, and return the one they used.
        response: types.GenerateResponse = running.generate(
            request=types.GenerateRequest(
                model=running.name, image=image, image_embedding=cached, prompt=prompt
            )
        )
        # None from YOLO-World, which drops another model's embedding too.
        _last_embedding = (
            None
            if response.image_embedding is None
            else (key, response.image_embedding)
        )
        return response.annotations


def _run(*, name: str, device: Optional[types.Device]) -> types.Model:
    # The running model for name and device, loading it if needed; under _lock.
    global running_model, _running_device

    model_type = _registry.get_model_type_by_name(name=name)
    model: types.Model | None = running_model
    if model is None or model.name != model_type.name or _running_device != device:
        # Release the previous model first, so two models are never loaded at once.
        model = running_model = None
        model = running_model = model_type(device=device)
        _running_device = device
    return model
