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
    np.testing.assert_array_equal(_tracker._to_bfloat16(x), [one, one, one + 2 * step])


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
        _video._component_areas(mask),
        [
            [2, 0, 0, 2, 2],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 1, 0],
        ],
    )


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
