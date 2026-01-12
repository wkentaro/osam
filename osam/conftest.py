from typing import Final

from loguru import logger

import osam


def pytest_configure(config):
    config.addinivalue_line("markers", "extra: marks tests as extra")

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
        if default_test_models_only and model not in DEFAULT_TEST_MODELS:
            continue
        model.pull()
    logger.info("Finished pulling all registered model types")
