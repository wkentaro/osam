import importlib.metadata

__version__ = importlib.metadata.version("osam")

try:
    import onnxruntime  # noqa: F401
except ImportError as e:
    raise ImportError(
        "osam requires onnxruntime: pip install 'osam[cpu]' or 'osam[gpu]'"
    ) from e

from . import apis
from . import types
from ._client import Client
from ._client import ResponseError

__all__ = [
    "Client",
    "ResponseError",
    "apis",
    "types",
]
