import os
import pathlib

import numpy as np
import PIL.Image
import pytest

from osam import _requests
from osam import _runtime
from osam import _types as types
from osam._models.sam3 import _timeline


class _Predictor:
    obj_ids = np.array([], np.int64)

    def __init__(self, *args) -> None:
        pass

    def add_prompt(self, *args, **kwargs) -> None:
        pass

    def propagate_in_video(self, **kwargs) -> list:
        return []


def test_a_null_threshold_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cache keys an action by its JSON, where an omitted threshold is
    # null, so a null threshold must pass as the omitted one does.
    monkeypatch.setattr(_timeline, "VideoInference", _Predictor)
    prompts = [
        types.PromptAction(type="prompt", frame=0, texts=["dog"]),
        types.PromptAction(type="prompt", frame=0, texts=["dog"], score_threshold=None),
    ]
    prompts.append(types.PromptAction.model_validate_json(prompts[0].model_dump_json()))
    for prompt in prompts:
        timeline = _timeline.Timeline(
            sessions={}, get_features=lambda frame: {}, num_frames=1, height=4, width=4
        )
        timeline.apply(action=prompt)
    with pytest.raises(ValueError, match="no thresholds"):
        timeline.apply(
            action=types.PromptAction(
                type="prompt", frame=0, texts=["dog"], score_threshold=0.5
            )
        )


# ponytail: runs only with the video models exported locally, until they are
# uploaded as blobs.
@pytest.mark.skipif(
    not os.environ.get("OSAM_SAM3_VIDEO_MODELS"),
    reason="needs OSAM_SAM3_VIDEO_MODELS",
)
def test_replay_gives_the_same_result() -> None:
    image = np.asarray(
        PIL.Image.open(
            pathlib.Path(__file__).resolve().parents[4] / "examples/_images/dogs.jpg"
        ).convert("RGB")
    )
    # The dogs, shifted right a little on each frame.
    frames = [np.roll(image, 20 * i, axis=1) for i in range(3)]
    height, width = image.shape[:2]
    prompt = {"type": "prompt", "frame": 0, "texts": ["dog"]}

    def track(runtime: _runtime.Runtime, actions: list) -> _requests.TrackResponse:
        return runtime.track(
            request=_requests.TrackRequest(
                model="sam3", frames=frames, actions=actions, keep_alive=-1
            )
        )

    runtime = _runtime.Runtime(device="cpu")
    detected = track(runtime, [prompt])
    assert len(detected.done.objects) >= 2
    first = detected.done.objects[0].object_id
    refine = {
        "type": "refine",
        "frame": 1,
        "object_id": first,
        "points": [[width / 2, height / 2]],
        "point_labels": [0],
        "base": detected.done.result,
    }
    refined = track(runtime, [prompt, refine])
    assert refined.done.cache == "partial"
    remove = {"type": "remove", "object_id": first, "base": refined.done.result}
    removed = track(runtime, [prompt, refine, remove])
    assert first not in [o.object_id for o in removed.done.objects]
    runtime.close()

    replayed = track(_runtime.Runtime(device="cpu"), [prompt, refine, remove])
    assert replayed.done.cache == "miss"
    assert replayed.done.result == removed.done.result


class _Session:
    _provider: str

    def __init__(self, *, provider: str) -> None:
        self._provider = provider

    def get_providers(self) -> list[str]:
        return [self._provider, "CPUExecutionProvider"]


@pytest.mark.parametrize(
    "provider", ["CUDAExecutionProvider", "WebGpuExecutionProvider"]
)
def test_video_models_fail_off_the_image_encoders_device(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from osam import _onnx
    from osam._models.sam3 import _models

    # onnxruntime dropped the provider, leaving the CPU.
    monkeypatch.setattr(
        _onnx,
        "_create_session",
        lambda path, providers: _Session(provider="CPUExecutionProvider"),
    )
    monkeypatch.setenv("OSAM_SAM3_VIDEO_MODELS", "/models")
    model = _models.Sam3.__new__(_models.Sam3)
    model._inference_sessions = {"image_encoder": _Session(provider=provider)}
    with pytest.raises(RuntimeError, match=f"{provider} failed to load"):
        model.new_timeline(
            get_embedding=lambda frame: None,
            num_frames=1,
            height=4,
            width=4,
        )
    assert model._video_sessions is None
