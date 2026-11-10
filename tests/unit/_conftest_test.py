from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import conftest


@pytest.mark.parametrize("markexpr", ["", "not extra"])
def test_pull_models_pulls_missing_selected_models(
    monkeypatch: pytest.MonkeyPatch, markexpr: str
) -> None:
    models = [
        SimpleNamespace(
            name=name,
            pull=Mock(),
            get_size=Mock(return_value=1 if name == "sam2:tiny" else None),
        )
        for name in ["efficientsam:10m", "sam3:latest", "sam2:tiny", "sam:100m"]
    ]
    monkeypatch.setattr(conftest.osam._registry, "registered_model_types", models)
    config = SimpleNamespace(
        option=SimpleNamespace(collectonly=False, markexpr=markexpr)
    )

    conftest.pull_models(config=config, nodeids=["tests/unit/_json_test.py::test_a"])
    assert not any(model.pull.called for model in models)

    conftest.pull_models(config=config, nodeids=["tests/e2e/cli_test.py::test_run"])
    assert [model.name for model in models if model.pull.called] == (
        ["efficientsam:10m", "sam3:latest"]
        if markexpr
        else ["efficientsam:10m", "sam3:latest", "sam:100m"]
    )
