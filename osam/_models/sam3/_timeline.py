# Applies /api/track's actions to SAM3's video predictor, as ADR 0005 defines
# them: each action applies to the result of the earlier ones propagated over
# every frame, in both directions from the frame it names.
import hashlib
import typing
from collections.abc import Callable

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from osam import _types as types
from osam._types._prompt import THRESHOLDS
from osam._types._prompt import PointLabel

from ._tracker import IMAGE_SIZE
from ._video import VideoInference


class Timeline:
    _video: typing.Final[VideoInference]
    _num_frames: typing.Final[int]
    _scale: typing.Final[NDArray[np.float32]]
    # The box and points the user gave each object on each frame, which a
    # refinement adds to, as the predictor replaces them.
    _prompts: dict[int, dict[int, tuple[NDArray | None, list, list]]]
    # The annotations on each frame, by object id, and their digest, kept
    # in place of the predictor's full-frame masks.
    _annotations: list[list[types.Annotation]]
    _result: str

    def __init__(
        self,
        *,
        sessions: dict[str, onnxruntime.InferenceSession],
        get_features: Callable[[int], dict[str, types.Feature]],
        num_frames: int,
        height: int,
        width: int,
    ):
        self._video = VideoInference(sessions, get_features, num_frames, height, width)
        self._num_frames = num_frames
        self._scale = np.array([width, height], np.float32)
        self._prompts = {}
        self._annotations = [[]] * num_frames
        self._result = ""

    def annotations(self, *, frame: int) -> list[types.Annotation]:
        return self._annotations[frame]

    def result(self) -> str:
        # The digest of the object ids and masks on every frame, as base.
        return self._result

    def apply(self, *, action: types.Action) -> None:
        # The first action is the prompt, and only the first. Raises
        # ValueError for an action the result doesn't allow, such as a
        # refinement of an object that isn't there.
        if isinstance(action, types.PromptAction):
            self._check_frame(frame=action.frame)
            if action.texts is not None:
                self._prompt_texts(action=action)
                return
            if action.example_boxes is not None:
                raise ValueError("example_boxes need texts")
            if action.points is None and action.box is None:
                raise ValueError("a prompt takes points, a box, or texts")
            self._start_without_detection()
            self._add(
                frame=action.frame,
                points=action.points,
                point_labels=action.point_labels,
                box=action.box,
            )
        elif isinstance(action, types.AddAction):
            self._check_frame(frame=action.frame)
            self._add(
                frame=action.frame,
                points=action.points,
                point_labels=action.point_labels,
                box=action.box,
            )
        elif isinstance(action, types.RefineAction):
            self._check_frame(frame=action.frame)
            self._check_object(object_id=action.object_id)
            box, points, labels = self._prompts.setdefault(action.object_id, {}).get(
                action.frame, (None, [], [])
            )
            self._prompts[action.object_id][action.frame] = (
                box,
                [*points, *action.points],
                [*labels, *action.point_labels],
            )
            self._prompt_object(object_id=action.object_id, frame=action.frame)
        else:
            self._check_object(object_id=action.object_id)
            self._video.remove_object(action.object_id)
            self._fetch()

    def _check_frame(self, *, frame: int) -> None:
        if frame >= self._num_frames:
            raise ValueError(f"frame {frame} is past the last of {self._num_frames}")

    def _check_object(self, *, object_id: int) -> None:
        if object_id not in self._video.obj_ids.tolist():
            raise ValueError(f"no object {object_id}")

    def _prompt_texts(self, *, action: types.PromptAction) -> None:
        # A null threshold is unset, as the action's cache key has it.
        if any(getattr(action, name) is not None for name in THRESHOLDS):
            raise ValueError("track takes no thresholds")
        texts: list[str] = typing.cast(list[str], action.texts)
        if len(texts) != 1:
            raise ValueError("track takes one text")
        boxes_xywh: NDArray | None = None
        box_labels: NDArray | None = None
        if action.example_boxes is not None:
            if len(action.example_boxes) != 1:
                raise ValueError("track takes at most one example box")
            x1, y1, x2, y2 = action.example_boxes[0]
            boxes_xywh = np.array([x1, y1, x2 - x1, y2 - y1], np.float32) / np.tile(
                self._scale, 2
            )
            box_labels = np.array(action.example_box_labels or [1])
        self._video.add_prompt(
            action.frame,
            text_str=texts[0],
            boxes_xywh=boxes_xywh,
            box_labels=box_labels,
        )
        self._propagate(frame=action.frame)

    def _start_without_detection(self) -> None:
        # As if a text prompt had found nothing, so points and boxes add
        # objects that only the tracker follows.
        video = self._video
        video.cached_frame_outputs = {i: {} for i in range(self._num_frames)}
        video.frames_with_outputs = set(range(self._num_frames))
        video._add_action_history("propagation_full", 0, None)
        video._add_action_history("propagation_full", self._num_frames - 1, None, True)

    def _add(
        self,
        *,
        frame: int,
        points: list[tuple[float, float]] | None,
        point_labels: list[PointLabel] | None,
        box: tuple[float, float, float, float] | None,
    ) -> None:
        object_id: int = self._video.max_obj_id + 1
        self._prompts[object_id] = {
            frame: (
                None if box is None else np.array(box, np.float32),
                list(points or []),
                list(point_labels or []),
            )
        }
        self._prompt_object(object_id=object_id, frame=frame)

    def _prompt_object(self, *, object_id: int, frame: int) -> None:
        box, points, labels = self._prompts[object_id][frame]

        def normalize(xy: NDArray[np.float32]) -> NDArray:
            # Pixel centers, as the point decoder reads them at IMAGE_SIZE.
            return (
                types.scale_points(
                    points=xy.reshape(-1, 2), scale=IMAGE_SIZE / self._scale
                )
                / IMAGE_SIZE
            )

        self._video.add_tracker_new_points(
            frame,
            object_id,
            points=normalize(np.array(points, np.float32)) if points else None,
            labels=np.array(labels, np.int32) if points else None,
            box=None if box is None else normalize(box).reshape(-1),
        )
        self._propagate(frame=frame)

    def _propagate(self, *, frame: int) -> None:
        for reverse in (False, True):
            for _ in self._video.propagate_in_video(
                start_frame_idx=frame, reverse=reverse
            ):
                pass
        self._fetch()

    def _fetch(self) -> None:
        # Every frame's outputs as the predictor has them now: hotstart can
        # remove an object from frames the other direction already returned,
        # and a removed object no longer claims its pixels from the others.
        digest = hashlib.sha256()
        for frame, out in self._video.propagate_in_video(start_frame_idx=0):
            self._annotations[frame] = []
            for object_id, score, mask in zip(
                out["out_obj_ids"].tolist(),
                out["out_probs"].tolist(),
                out["out_binary_masks"],
            ):
                digest.update(f"{frame} {object_id}\n".encode())
                digest.update(np.packbits(mask).tobytes())
                self._annotations[frame] += [
                    annotation.model_copy(update={"object_id": object_id})
                    for annotation in types.annotations_from_mask(
                        mask=mask, score=score
                    )
                ]
        self._result = f"sha256:{digest.hexdigest()}"
