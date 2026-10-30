from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests import conftest


@pytest.mark.parametrize("markexpr", ["", "not extra"])
def test_pytest_configure_pulls_selected_models(
    monkeypatch: pytest.MonkeyPatch, markexpr: str
) -> None:
    models = [
        SimpleNamespace(name=name, pull=Mock())
        for name in ["efficientsam:10m", "sam3:latest", "sam2:tiny", "sam:100m"]
    ]
    monkeypatch.setattr(conftest.osam.apis, "registered_model_types", models)
    config = SimpleNamespace(
        option=SimpleNamespace(collectonly=False, markexpr=markexpr),
        addinivalue_line=Mock(),
    )

    conftest.pytest_configure(config)

    assert [model.name for model in models if model.pull.called] == [
        model.name for model in (models[:3] if markexpr else models)
    ]
