import os
from typing import Final

import pytest
from loguru import logger

import osam
import osam._registry


def pytest_configure(config):
    config.addinivalue_line("markers", "extra: marks tests as extra")
    # A developer's OSAM_HOST would point the clients and the CLI under test at
    # their server. Here, as module-scoped fixtures start before monkeypatch.
    os.environ.pop("OSAM_HOST", None)

    if config.option.collectonly:
        return
    if _is_slave := hasattr(config, "workerinput"):  # noqa: F841
        return

    default_test_models_only: bool = "not extra" in config.option.markexpr
    DEFAULT_TEST_MODELS: Final[list[str]] = [
        "efficientsam:10m",
        "sam3:latest",
        "sam2:tiny",
    ]

    # Workers must not download the same file in parallel. A model already on
    # disk is verified when a test loads it, not here on every run.
    for model in osam._registry.registered_model_types:
        if default_test_models_only and model.name not in DEFAULT_TEST_MODELS:
            continue
        if model.get_size() is None:
            logger.info("Pulling {name}", name=model.name)
            model.pull()


@pytest.hookimpl(tryfirst=True)  # before xdist reads the groups
def pytest_collection_modifyitems(items):
    # Two SAM3 models don't fit in a CI runner's memory, so one worker runs them.
    for item in items:
        if "sam3" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("sam3"))
