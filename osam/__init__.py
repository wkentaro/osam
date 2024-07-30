import importlib.metadata

__version__ = importlib.metadata.version("osam")

from . import apis  # noqa: F401
from . import types  # noqa: F401
