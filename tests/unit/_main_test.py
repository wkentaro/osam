import contextlib
import json
import os
import pathlib
import subprocess
import sys
import types
import typing
from typing import Any

import numpy as np
import PIL.Image
import pytest
from click.testing import CliRunner

import osam.types
from osam import __main__
from osam import _registry
from osam import _runtime
from osam import _server
from osam import _types


def test_full_name_comes_from_the_server_list() -> None:
    models = [types.SimpleNamespace(name=n) for n in ["sam3:latest", "sam2:tiny"]]
    client = typing.cast(Any, types.SimpleNamespace(list=lambda: models))
    assert __main__._full_name(client=client, name="sam3") == "sam3:latest"
    assert __main__._full_name(client=client, name="sam2:tiny") == "sam2:tiny"
    # A name the server doesn't list stays as given.
    assert __main__._full_name(client=client, name="sam2") == "sam2"


def test_serve_reads_settings_from_flags_then_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings: list[dict[str, Any]] = []

    class Runtime:
        def __init__(self, **kwargs: Any) -> None:
            settings.append(kwargs)

        def close(self) -> None:
            pass

    monkeypatch.setattr(_runtime, "Runtime", Runtime)
    monkeypatch.setattr(_server, "create_app", lambda *, runtime: None)
    monkeypatch.setattr(
        _server,
        "uvicorn_server",
        lambda **config: types.SimpleNamespace(run=lambda: None),
    )
    env = {"OSAM_SERVE_DEVICE": "cpu", "OSAM_MAX_LOADED_MODELS": "2"}

    for args, env_vars, expected in [
        ([], {}, {"device": None, "max_loaded_models": 1}),
        ([], env, {"device": "cpu", "max_loaded_models": 2}),
        (["--max-loaded-models", "3"], env, {"device": "cpu", "max_loaded_models": 3}),
    ]:
        result = CliRunner(env=env_vars).invoke(__main__.serve, args)
        assert result.exit_code == 0, result.output
        assert settings.pop() == expected
    result = CliRunner(env={"OSAM_MAX_LOADED_MODELS": "0"}).invoke(__main__.serve)
    assert result.exit_code == 2
    assert (
        "OSAM_MAX_LOADED_MODELS"
        in CliRunner().invoke(__main__.serve, ["--help"]).output
    )


def test_run_converts_the_image_to_rgb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[:, :3] = (255, 0, 0)
    rgb[:, 3:] = (0, 0, 255)
    palette_path = tmp_path / "palette.png"
    PIL.Image.fromarray(rgb).quantize(colors=2).save(palette_path)
    rgba_path = tmp_path / "rgba.png"
    rgba = np.dstack([rgb, np.full((4, 6), 255, dtype=np.uint8)])
    PIL.Image.fromarray(rgba).save(rgba_path)

    received: list[Any] = []

    class Client:
        def __init__(self, device: str | None = None) -> None:
            pass

        def __enter__(self) -> "Client":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def segment(self, model: str, image: Any, **prompt: Any) -> list[Any]:
            received.append(image)
            return [
                osam.types.Annotation(
                    bounding_box=osam.types.BoundingBox(xmin=0, ymin=0, xmax=1, ymax=1),
                    mask=np.ones((2, 2), dtype=bool),
                )
            ]

    monkeypatch.setattr(__main__, "Client", Client)

    for path in [palette_path, rgba_path]:
        result = CliRunner().invoke(__main__.run, ["fake", "--image", str(path)])
        assert result.exit_code == 0, result.output
        np.testing.assert_array_equal(received.pop(), rgb)
        assert result.stdout_bytes.startswith(b"\x89PNG")


def test_log_file_has_no_color_codes(tmp_path: pathlib.Path) -> None:
    # A subprocess, as the cli group replaces this process's log handlers.
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    result = subprocess.run(
        [sys.executable, "-m", "osam", "help", "nosuch"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    log: str = (tmp_path / ".cache/osam/osam.log").read_text()
    assert "Unknown subcommand 'nosuch'" in log
    assert "\x1b[" not in log


def test_cli_runs_twice_in_one_process(tmp_path: pathlib.Path) -> None:
    # A subprocess, as the cli group replaces this process's log handlers.
    env = {**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    script = (
        "from click.testing import CliRunner\n"
        "from osam.__main__ import cli\n"
        "for _ in range(2):\n"
        "    result = CliRunner().invoke(cli, ['help'])\n"
        "    assert result.exit_code == 0, repr(result.exception)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr


def test_run_video_writes_a_png_per_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from tests.unit._client_test import _serving
    from tests.unit._server_test import _patch_tracking

    _patch_tracking(
        monkeypatch=monkeypatch,
        model_type=_registry.get_model_type_by_name(name="efficientsam:10m"),
    )
    video = tmp_path / "video"
    video.mkdir()
    for i in [10, 2, 1]:  # name order takes 2 before 10
        PIL.Image.fromarray(np.full((4, 4, 3), i, np.uint8)).save(video / f"{i}.jpg")
    for name in [".DS_Store", "1.json"]:  # files beside the frames are no frames
        (video / name).write_text("{}")
    out = tmp_path / "out"
    with _serving() as host:
        monkeypatch.setenv("OSAM_HOST", host)
        result = CliRunner().invoke(
            __main__.run,
            ["efficientsam:10m", "--video", str(video), "--point", "0,0"]
            + ["-o", str(out), "--json"],
        )
    assert result.exit_code == 0, result.output
    *frames, done = (json.loads(line) for line in result.stdout.splitlines())
    assert "Preparing video" in result.stderr
    assert "Tracking action 1, frame 3/3" in result.stderr
    assert [f["frame"] for f in frames] == [0, 1, 2]
    assert done["objects"] == [{"object_id": 0, "first_frame": 0, "last_frame": 2}]
    assert sorted(p.name for p in out.iterdir()) == ["1.png", "10.png", "2.png"]
    assert PIL.Image.open(out / "10.png").size == (4, 4)

    result = CliRunner().invoke(
        __main__.run, ["efficientsam:10m", "--video", str(video)]
    )
    assert result.exit_code == 2
    assert "--video needs -o" in result.output

    # Also without case or Unicode form, as -o writes one file for each pair
    # on macOS.
    for names in [["1.png"], ["a.jpg", "A.png"], ["\u00e9.jpg", "e\u0301.png"]]:
        for name in names:
            PIL.Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(video / name)
        result = CliRunner().invoke(
            __main__.run,
            ["efficientsam:10m", "--video", str(video), "--point", "0,0"]
            + ["-o", str(out)],
        )
        assert result.exit_code == 2
        assert "same name" in result.output
        for name in names:
            (video / name).unlink()


def test_an_object_keeps_its_color_when_an_earlier_one_leaves() -> None:
    def annotation(*, object_id: int, x: int) -> _types.Annotation:
        mask = np.zeros((8, 8), bool)
        mask[2:6, x : x + 4] = True
        (a,) = _types.annotations_from_mask(mask=mask, score=1)
        return a.model_copy(update={"object_id": object_id})

    image = np.zeros((8, 8, 3), np.uint8)
    both = __main__._visualize_objects(
        image=image,
        annotations=[annotation(object_id=0, x=0), annotation(object_id=1, x=4)],
    )
    alone = __main__._visualize_objects(
        image=image, annotations=[annotation(object_id=1, x=4)]
    )
    assert both[4, 5].tolist() == alone[4, 5].tolist() != [0, 0, 0]


def test_run_video_passes_include_masks_to_the_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    from tests.unit._client_test import _serving
    from tests.unit._server_test import _patch_tracking

    _patch_tracking(
        monkeypatch=monkeypatch,
        model_type=_registry.get_model_type_by_name(name="efficientsam:10m"),
    )
    video = tmp_path / "video"
    video.mkdir()
    PIL.Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(video / "0.jpg")
    prompt = '{"points": [[0, 0]], "point_labels": [1], "include_masks": false}'
    with _serving() as host:
        monkeypatch.setenv("OSAM_HOST", host)
        result = CliRunner().invoke(
            __main__.run,
            ["efficientsam:10m", "--video", str(video), "--prompt", prompt, "--json"]
            + ["-o", str(tmp_path / "out")],
        )
    assert result.exit_code == 0, result.output
    frame, _ = (
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    )
    (annotation,) = frame["annotations"]
    assert annotation["mask"] is None
    assert annotation["bounding_box"] is not None


@pytest.mark.parametrize(
    "alias",
    [
        "same directory",
        pytest.param(
            "symlink",
            marks=pytest.mark.skipif(
                sys.platform == "win32",
                reason="directory symlinks require privileges on Windows",
            ),
        ),
        "hardlink",
    ],
)
@pytest.mark.parametrize("overwrite", [False, True])
def test_video_output_preserves_source_frames(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, alias: str, overwrite: bool
) -> None:
    video = tmp_path / "frames"
    video.mkdir()
    source = video / "0.png"
    PIL.Image.fromarray(np.full((4, 4, 3), 80, np.uint8)).save(source)
    original = source.read_bytes()
    out = video
    if alias == "symlink":
        out = tmp_path / "alias"
        out.symlink_to(video, target_is_directory=True)
    elif alias == "hardlink":
        out = tmp_path / "out"
        out.mkdir()
        (out / "0.png").hardlink_to(source)

    def unexpected(**kwargs):
        raise AssertionError("started inference before checking output paths")

    monkeypatch.setattr(__main__, "_client", unexpected)
    args = ["sam3", "--video", str(video), "--text", "dog", "-o", str(out)]
    result = CliRunner().invoke(
        __main__.run, args + (["--overwrite"] if overwrite else [])
    )
    assert result.exit_code == 2, result.output
    assert "input frame" in result.output
    assert source.read_bytes() == original


@pytest.mark.parametrize("alias_during_inference", [False, True])
def test_video_output_requires_explicit_overwrite(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_during_inference: bool,
) -> None:
    video, out = tmp_path / "frames", tmp_path / "out"
    video.mkdir()
    out.mkdir()
    source, target = video / "0.png", out / "0.png"
    PIL.Image.fromarray(np.full((4, 4, 3), 80, np.uint8)).save(source)
    original = source.read_bytes()
    target.write_bytes(b"existing output")
    calls = []

    def track(*args, **kwargs):
        calls.append(True)
        if alias_during_inference:
            target.unlink()
            target.hardlink_to(source)
        return osam.types.TrackResponse(
            frames=[osam.types.TrackFrame(frame=0, annotations=[])],
            done=osam.types.TrackDone(
                objects=[], result="sha256:" + "0" * 64, cache="miss", images=[]
            ),
        )

    monkeypatch.setattr(
        __main__,
        "_client",
        lambda **kwargs: contextlib.nullcontext(types.SimpleNamespace(track=track)),
    )
    monkeypatch.setattr(
        __main__,
        "_visualize_objects",
        lambda **kwargs: np.full((4, 4, 3), 255, np.uint8),
    )
    args = ["sam3", "--video", str(video), "--text", "dog", "-o", str(out)]
    result = CliRunner().invoke(__main__.run, args)
    assert result.exit_code == 2, result.output
    assert "--overwrite" in result.output
    assert not calls
    assert target.read_bytes() == b"existing output"

    result = CliRunner().invoke(__main__.run, args + ["--overwrite"])
    if alias_during_inference:
        assert result.exit_code == 2, result.output
        assert "input frame" in result.output
    else:
        assert result.exit_code == 0, result.output
        assert np.asarray(PIL.Image.open(target)).min() == 255
    assert calls == [True]
    assert source.read_bytes() == original


def test_run_rejects_device_with_osam_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setenv("OSAM_HOST", "127.0.0.1:1")
    image = tmp_path / "image.png"
    PIL.Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(image)
    result = CliRunner().invoke(
        __main__.run, ["efficientsam:10m", "--image", str(image), "--device", "cpu"]
    )
    assert result.exit_code == 2  # noqa: PLR2004
    assert "unset OSAM_HOST" in result.output
