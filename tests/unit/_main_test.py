import pathlib
import types
import typing
from typing import Any

import numpy as np
import PIL.Image
import pytest
from click.testing import CliRunner

import osam.types
from osam import __main__
from osam import _runtime
from osam import _server


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
