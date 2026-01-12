def pytest_configure(config):
    config.addinivalue_line("markers", "extra: marks tests as extra")
