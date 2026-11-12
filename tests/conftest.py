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


def pull_models(*, config, nodeids) -> None:
    # Before any test runs, so that workers don't download the same file in
    # parallel. A model already on disk is verified when a test loads it.

    # The tests that load real weights. A file missing here still passes: its
    # workers each download the model.
    WEIGHT_TESTS: Final = ("e2e/", "_client_test.py", "_models/_segment_test.py")
    DEFAULT_TEST_MODELS: Final = ("efficientsam:10m", "sam3:latest", "sam2:tiny")

    if config.option.collectonly:
        return
    if not any(name in nodeid for nodeid in nodeids for name in WEIGHT_TESTS):
        return
    default_test_models_only: bool = "not extra" in config.option.markexpr
    for model in osam._registry.registered_model_types:
        if default_test_models_only and model.name not in DEFAULT_TEST_MODELS:
            continue
        if model.get_size() is None:
            logger.info("Pulling {name}", name=model.name)
            model.pull()


def pytest_collection_finish(session):
    # Without xdist. With it, a worker must not pull, and the hook below does.
    if not hasattr(session.config, "workerinput"):
        pull_models(
            config=session.config, nodeids=[item.nodeid for item in session.items]
        )


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids):
    # On the controller, once per worker, before it schedules any test.
    pull_models(config=node.config, nodeids=ids)


@pytest.hookimpl(tryfirst=True)  # before xdist reads the groups
def pytest_collection_modifyitems(items):
    # Two SAM3 models don't fit in a CI runner's memory, so one worker runs the
    # tests that load it. Those under _models/sam3/ load no weights, or mark
    # themselves.
    for item in items:
        if "sam3" in item.nodeid and "/_models/sam3/" not in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("sam3"))
