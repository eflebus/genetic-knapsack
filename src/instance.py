from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    capacity: int
    num_items: int
    items_weights: np.ndarray
    items_profits: np.ndarray
    optimum: int
    optimal_selection: np.ndarray  # Array of -1s for type A instances
