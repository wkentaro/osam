def pytest_configure(config):
    config.addinivalue_line("markers", "heavy: marks tests as heavy (large models)")
