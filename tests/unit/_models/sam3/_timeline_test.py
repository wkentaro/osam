import contextlib
import functools
import pathlib
import time
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
from loguru import logger

from osam import _requests
from osam import _runtime
from osam import _types
from osam import types
from osam._models.sam3 import _timeline
from osam._models.sam3 import _tracker
from osam._models.sam3._models import Sam3
from tests.unit import _fakes


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
        _types.PromptAction(type="prompt", frame=0, texts=["dog"]),
        _types.PromptAction(
            type="prompt", frame=0, texts=["dog"], score_threshold=None
        ),
    ]
    prompts.append(
        _types.PromptAction.model_validate_json(prompts[0].model_dump_json())
    )
    for prompt in prompts:
        timeline = _timeline.Timeline(
            sessions={}, get_features=lambda frame: {}, num_frames=1, height=4, width=4
        )
        timeline.apply(action=prompt)
    with pytest.raises(ValueError, match="no thresholds"):
        timeline.apply(
            action=_types.PromptAction(
                type="prompt", frame=0, texts=["dog"], score_threshold=0.5
            )
        )


def test_replay_gives_the_same_result() -> None:
    # Only neural inference is synthetic. Runtime, Timeline, VideoInference,
    # and Tracker still apply, propagate, hash, cache, and replay every action.
    size = _tracker.LOW_RES_MASK_SIZE
    masks = np.full((2, 1, size, size), -1, np.float32)
    masks[0, :, 40:240, 20:120] = 1
    masks[1, :, 40:240, 170:270] = 1

    def session(outputs: list[str], infer: Callable, *, batch: int | str = "objects"):
        def run(names, inputs):
            result = infer(inputs)
            return [result[n] for n in (outputs if names is None else names)]

        return SimpleNamespace(
            get_outputs=lambda: [SimpleNamespace(name=n) for n in outputs],
            get_inputs=lambda: [SimpleNamespace(name="box_coords", shape=[batch])],
            run=run,
        )

    def decode(inputs: dict) -> dict:
        batch = len(inputs["image_embeddings"])
        logits = _tracker._resize(inputs["image_embeddings"], size, size)
        if inputs["has_mask_input"].item():
            logits = inputs["mask_input"].copy()
        # A negative refinement removes pixels, not just an action counter.
        for i, labels in enumerate(inputs["point_labels"]):
            for point, label in zip(inputs["point_coords"][i], labels):
                if label == 0:
                    x, y = (point * size / _tracker.IMAGE_SIZE).astype(int)
                    logits[i, :, y - 5 : y + 5, x - 5 : x + 5] = -1
        return {
            "single_low_res_masks": logits,
            "low_res_masks": np.repeat(logits, 3, axis=1),
            "single_iou_predictions": np.full((batch, 1), 0.9, np.float32),
            "iou_predictions": np.full((batch, 3), 0.9, np.float32),
            "single_obj_ptr": np.zeros((batch, 256), np.float32),
            "obj_ptrs": np.zeros((batch, 3, 256), np.float32),
            "object_score_logits": np.full((batch, 1), 10, np.float32),
        }

    sessions = {
        "language_encoder": session(
            ["language_mask", "language_features", "language_embeddings"],
            lambda inputs: dict.fromkeys(
                ["language_mask", "language_features", "language_embeddings"],
                np.zeros((1, 1), np.float32),
            ),
        ),
        "decoder": session(
            ["scores", "mask_logits"],
            lambda inputs: {"scores": np.array([0.9, 0.8]), "mask_logits": masks},
            batch=1,
        ),
        "point_decoder": session(
            [
                "single_low_res_masks",
                "low_res_masks",
                "single_iou_predictions",
                "iou_predictions",
                "single_obj_ptr",
                "obj_ptrs",
                "object_score_logits",
            ],
            decode,
        ),
        "mask_downsampler": session(
            ["image_features", "mask_input", "no_obj_ptr"],
            lambda inputs: {
                "image_features": inputs["image_embeddings"],
                "mask_input": _tracker._resize(inputs["mask"], size, size),
                "no_obj_ptr": np.zeros((1, 256), np.float32),
            },
        ),
        "memory_encoder": session(
            ["maskmem_features"],
            lambda inputs: {
                "maskmem_features": _tracker._resize(inputs["mask_logits"], 8, 8)
            },
        ),
        "memory_attention": session(
            ["pix_feat_with_mem"],
            lambda inputs: {"pix_feat_with_mem": inputs["memory"][:1]},
        ),
    }

    class Model(_fakes.FakeModel):
        _inference_sessions: dict
        name = "sam3:latest"
        new_timeline = Sam3.new_timeline

        def __init__(self, device=None):
            super().__init__(device=device)
            self._inference_sessions = sessions

        @classmethod
        def tracks(cls) -> bool:
            return True

        def encode_image(self, *, image: np.ndarray) -> _types.ImageEmbedding:
            embedding = super().encode_image(image=image)
            return embedding.model_copy(
                update={"extra_features": [np.zeros((1, 1, 8, 8), np.float32)] * 6}
            )

    _check_replay(
        new_runtime=lambda: _runtime.Runtime(device="cpu", model_types=[Model]),
        frames=[np.full((32, 32, 3), i, np.uint8) for i in range(3)],
    )


# Real-weight numerical reproducibility, isolated from the OS test matrix.
@pytest.mark.model_conformance
@pytest.mark.xdist_group("sam3")
def test_sam3_replay_gives_the_same_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def measured(method):
        @functools.wraps(method)
        def call(*args, **kwargs):
            started = time.perf_counter()
            try:
                return method(*args, **kwargs)
            finally:
                logger.info(
                    "{} took {:.2f}s",
                    method.__qualname__,
                    time.perf_counter() - started,
                )

        return call

    for cls, name in [
        (Sam3, "__init__"),
        (Sam3, "encode_image"),
        (_timeline.Timeline, "apply"),
    ]:
        monkeypatch.setattr(cls, name, measured(getattr(cls, name)))

    image = np.asarray(
        PIL.Image.open(
            pathlib.Path(__file__).resolve().parents[4] / "examples/_images/dogs.jpg"
        ).convert("RGB")
    )
    # Two frames exercise temporal inference and cold replay. The synthetic
    # test above covers refinement in the middle of a three-frame video.
    frames = [image, np.roll(image, 20, axis=1)]
    _check_replay(
        new_runtime=lambda: _runtime.Runtime(device="cpu"),
        frames=frames,
    )


def _check_replay(
    *,
    new_runtime: Callable[[], _runtime.Runtime],
    frames: list[np.ndarray],
) -> None:
    prompt = {"type": "prompt", "frame": 0, "texts": ["dog"]}

    def track(runtime: _runtime.Runtime, actions: list) -> types.TrackResponse:
        started = time.perf_counter()
        response = runtime.track(
            request=_requests.TrackRequest(
                model="sam3", frames=[*frames], actions=actions, keep_alive=-1
            )
        )
        logger.info(
            "Track {} ({}) took {:.2f}s",
            [a["type"] for a in actions],
            response.done.cache,
            time.perf_counter() - started,
        )
        return response

    with contextlib.closing(new_runtime()) as runtime:
        detected = track(runtime, [prompt])
        assert detected.done.cache == "miss"
        assert len(detected.done.objects) >= 2
        first = detected.done.objects[0].object_id
        annotation = next(
            a for a in detected.frames[1].annotations if a.object_id == first
        )
        # A negative point inside the mask, nearest its center. A fixed image
        # coordinate can miss the object and make refinement a no-op.
        mask = annotation.mask
        assert mask is not None
        ys, xs = np.nonzero(mask)
        center = (
            (xs - mask.shape[1] / 2) ** 2 + (ys - mask.shape[0] / 2) ** 2
        ).argmin()
        refine_point = (
            float(annotation.bounding_box.xmin + xs[center]),
            float(annotation.bounding_box.ymin + ys[center]),
        )
        refine = {
            "type": "refine",
            "frame": 1,
            "object_id": first,
            "points": [refine_point],
            "point_labels": [0],
            "base": detected.done.result,
        }
        refined = track(runtime, [prompt, refine])
        assert refined.done.cache == "partial"
        assert refined.done.result != detected.done.result
        remove = {"type": "remove", "object_id": first, "base": refined.done.result}
        removed = track(runtime, [prompt, refine, remove])
        assert removed.done.objects
        assert first not in [o.object_id for o in removed.done.objects]
        assert removed.done.result != refined.done.result
        assert track(runtime, [prompt, refine, remove]).done.cache == "hit"

    with contextlib.closing(new_runtime()) as runtime:
        replayed = track(runtime, [prompt, refine, remove])
        assert replayed.done.cache == "miss"
        assert replayed.done.result == removed.done.result


def test_points_and_box_reach_the_tracker_as_pixel_centers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class Predictor(_Predictor):
        max_obj_id = 0

        def _add_action_history(self, *args) -> None:
            pass

        def add_tracker_new_points(self, *args, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(_timeline, "VideoInference", Predictor)
    timeline = _timeline.Timeline(
        sessions={}, get_features=lambda frame: {}, num_frames=1, height=128, width=256
    )

    timeline.apply(
        action=_types.PromptAction(
            type="prompt",
            frame=0,
            points=[(10, 20)],
            point_labels=[1],
            box=(0, 0, 255, 127),
        )
    )

    # The point decoder adds 0.5 to read an integer as a pixel center of the
    # image resized to IMAGE_SIZE.
    [call] = calls
    scale = _timeline.IMAGE_SIZE / np.array([256, 128])
    for sent, pixels in [
        (call["points"], [(10, 20)]),
        (call["box"].reshape(-1, 2), [(0, 0), (255, 127)]),
    ]:
        np.testing.assert_allclose(
            sent * _timeline.IMAGE_SIZE + 0.5,
            (np.array(pixels) + 0.5) * scale,
            rtol=1e-6,
        )
