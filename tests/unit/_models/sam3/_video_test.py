import gc
import types
import typing
import weakref

import numpy as np
import pytest

from osam._models.sam3 import _tracker
from osam._models.sam3 import _video


def test_empty_masks() -> None:
    empty = np.zeros((0, 4, 4), np.float32)
    masks = np.ones((2, 4, 4), np.float32)
    assert _video._mask_iou(empty > 0, masks > 0).shape == (0, 2)
    assert _video._mask_iou(masks > 0, empty > 0).shape == (2, 0)
    assert _video._nms_masks(np.zeros(0, np.float32), empty, 0.1).tolist() == []
    assert _tracker._resize(empty[:, None], 8, 6).shape == (0, 1, 8, 6)


def test_nms_masks() -> None:
    masks = np.zeros((3, 4, 4), np.float32)
    masks[:2, :2] = 1  # two duplicates
    masks[2, 2:] = 1
    probs = np.array([0.6, 0.9, 0.7], np.float32)
    assert _video._nms_masks(probs, masks, 0.1).tolist() == [1, 2]


def test_resize_is_bilinear_without_align_corners() -> None:
    x = np.array([[[[0, 1]]]], np.float32)
    # Pixel centers at -0.25 and 0.25 of an input pixel, clamped at the edges.
    np.testing.assert_allclose(_tracker._resize(x, 1, 4), [[[[0, 0.25, 0.75, 1]]]])
    np.testing.assert_allclose(_tracker._resize(x, 1, 1), [[[[0.5]]]])


def test_to_bfloat16_rounds_to_nearest_even() -> None:
    one = np.float32(1)
    step = np.float32(2**-7)  # bfloat16 has 7 mantissa bits
    x = np.array([one + step / 4, one + step / 2, one + 3 * step / 2], np.float32)
    np.testing.assert_array_equal(
        _tracker._from_bfloat16(_tracker._to_bfloat16(x)), [one, one, one + 2 * step]
    )


def test_is_argmax_takes_the_first_of_tied_maxima() -> None:
    scores = np.array([[1, 2, 3], [3, 2, 1], [0, 2, 3]], np.float32)
    np.testing.assert_array_equal(
        _video._is_argmax(scores),
        scores.argmax(0)[None] == np.arange(3)[:, None],
    )


def test_fixed_memory_attention_inputs_pad_and_mark_validity() -> None:
    inputs = _tracker._pad_memory_attention_inputs(
        memory=np.ones((2, 3), np.float32),
        memory_tpos=np.array([6, 0], np.int64),
        obj_ptrs=np.ones((1, 4), np.float32),
        obj_ptr_tpos=np.zeros(1, np.float32),
    )
    assert inputs["memory"].shape == (_tracker.FIXED_NUM_MEMORIES, 3)
    assert inputs["obj_ptrs"].shape == (_tracker.FIXED_NUM_OBJ_PTRS, 4)
    assert inputs["memory_valid"].tolist() == [1, 1] + [0] * (
        _tracker.FIXED_NUM_MEMORIES - 2
    )
    assert inputs["obj_ptr_valid"].sum() == 1
    # The most the tracker gives: 4 conditioning frames and 6 recent ones, and
    # their pointers, which an older copy of 7 and 16 had no room for.
    full = _tracker._pad_memory_attention_inputs(
        memory=np.ones((10, 3), np.float32),
        memory_tpos=np.arange(10, dtype=np.int64),
        obj_ptrs=np.ones((19, 4), np.float32),
        obj_ptr_tpos=np.zeros(19, np.float32),
    )
    assert full["memory_valid"].all() and full["obj_ptr_valid"].all()


def test_component_areas_are_8_connected() -> None:
    mask = np.array(
        [
            [1, 0, 0, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        bool,
    )
    np.testing.assert_array_equal(
        _video.component_areas(mask),
        [
            [2, 0, 0, 2, 2],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    )


def test_cached_masks_keep_every_pixel() -> None:
    # 3 x 5 pixels, which don't fill a whole number of bytes.
    video = object.__new__(_video.VideoInference)
    video.num_frames, video.video_height, video.video_width = 1, 3, 5
    video.reset_state()
    mask = np.random.default_rng(0).random((3, 5)) > 0.5
    video._cache_frame_outputs(0, {7: mask, 8: ~mask}, removed_obj_ids=[8])
    cached = video._cached_masks(0)
    assert list(cached) == [7]
    assert cached[7].dtype == np.bool_
    np.testing.assert_array_equal(cached[7], mask)


def test_propagation_history() -> None:
    video = object.__new__(_video.VideoInference)
    video.num_frames, video.video_height, video.video_width = 10, 4, 4
    video.reset_state()
    video.frames_with_outputs.add(0)
    video._add_action_history("remove", None, [0])
    assert video._parse_action_history_for_propagation() == ("propagation_full", None)

    # A propagation from frame 0 that stopped at frame 4
    video._add_action_history("propagation_full", 0, None)
    video.cached_frame_outputs.update({i: {} for i in range(5)})
    assert len(list(video.propagate_in_video(max_frame_num_to_track=4))) == 5
    with pytest.raises(RuntimeError):
        next(video.propagate_in_video(start_frame_idx=5))

    # Hotstart removes an object through _tracker_remove_object.
    video.cached_frame_outputs[0] = {7: np.ones((4, 4), np.bool_)}
    video._tracker_remove_object(7)
    assert video.cached_frame_outputs[0] == {}

    # Forward from the last frame covers one frame, so reverse runs in full.
    video.action_history = []
    video._add_action_history("propagation_full", 9, None)
    assert video._parse_action_history_for_propagation() == ("propagation_full", None)
    video._add_action_history("propagation_full", 9, None, reverse=True)
    assert video._parse_action_history_for_propagation()[0] == "propagation_fetch"


def test_dropping_the_predictor_frees_it_without_gc() -> None:
    # The runtime unloads a model by dropping its timeline. A reference cycle
    # through the trackers would keep the sessions until a gc pass.
    def session(*, outputs: list[str], batch: object = "num_objects") -> object:
        return types.SimpleNamespace(
            get_outputs=lambda: [types.SimpleNamespace(name=n) for n in outputs],
            get_inputs=lambda: [types.SimpleNamespace(shape=[batch])],
        )

    sessions = {
        name: session(outputs=["mask_logits", "single_obj_ptr"])
        for name in {*_video.MODEL_NAMES, *_tracker.MODEL_NAMES}
    }
    predictor = _video.VideoInference(
        typing.cast(dict, sessions), lambda frame: {}, 1, 4, 4
    )
    predictor.trackers.append(predictor._new_tracker())
    ref = weakref.ref(predictor)
    gc.disable()
    try:
        del predictor
        assert ref() is None
    finally:
        gc.enable()


def test_progress_cancels_inside_the_hotstart_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = object.__new__(_video.VideoInference)
    video.num_frames = 100
    ran: list[int] = []

    def infer(frame: int, reverse: bool) -> dict:
        ran.append(frame)
        return {"removed_obj_ids": set()}

    def progress(frame: int) -> None:
        if frame == 2:
            raise RuntimeError("cancel")

    monkeypatch.setattr(video, "_run_single_frame_inference", infer)
    # No result has been yielded yet, but cancellation must already take effect.
    with pytest.raises(RuntimeError, match="cancel"):
        next(video._propagate_full(0, None, False, progress))
    assert ran == [0, 1]
