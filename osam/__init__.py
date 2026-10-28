import importlib.metadata

__version__ = importlib.metadata.version("osam")

try:
    import onnxruntime  # noqa: F401
except ImportError as e:
    raise ImportError(
        "osam requires onnxruntime: pip install 'osam[cpu]' or 'osam[gpu]'"
    ) from e

from . import apis  # noqa: F401
from . import types  # noqa: F401
