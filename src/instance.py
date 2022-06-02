from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InstanceData:
    capacity: int
    num_items: int
    weights: np.ndarray
    profits: np.ndarray
    optimum: int
    optimal_selection: np.ndarray  # Array of -1s for type A instances
