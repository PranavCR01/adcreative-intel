import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")


# Required by pytest-asyncio ≥ 0.21 to avoid per-test decorator boilerplate.
# Equivalent to [pytest] asyncio_mode = auto in pytest.ini / pyproject.toml.
def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            import inspect
            if inspect.iscoroutinefunction(item.function):
                item.add_marker(pytest.mark.asyncio)
