import importlib.metadata
import logging

__version__ = importlib.metadata.version("osam")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s [%(levelname)s] %(module)s:%(funcName)s:%(lineno)s - %(message)s"
    )
)
logger.addHandler(handler)
