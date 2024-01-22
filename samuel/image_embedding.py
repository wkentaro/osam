import dataclasses

import numpy as np


@dataclasses.dataclass
class ImageEmbedding:
    original_height: int
    original_width: int
    embedding: np.ndarray
