from typing import Final

import pytest
from loguru import logger

import osam


def pytest_configure(config):
    config.addinivalue_line("markers", "extra: marks tests as extra")
    config.addinivalue_line(
        "markers", "extended: additional inference cases beyond platform smoke tests"
    )

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

    # download should not happen in parallel
    logger.info("Pulling all registered model types")
    for model in osam.apis.registered_model_types:
        if default_test_models_only and model.name not in DEFAULT_TEST_MODELS:
            continue
        model.pull()
    logger.info("Finished pulling all registered model types")


@pytest.hookimpl(tryfirst=True)  # before xdist reads the groups
def pytest_collection_modifyitems(items):
    # Two SAM3 models don't fit in a CI runner's memory, so one worker runs them.
    for item in items:
        if "sam3" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("sam3"))
