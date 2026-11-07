import types
import typing
from typing import Any

import pytest
from click.testing import CliRunner

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
