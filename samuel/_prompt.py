import dataclasses

import numpy as np


@dataclasses.dataclass
class Prompt:
    points: np.ndarray
    point_labels: np.ndarray
