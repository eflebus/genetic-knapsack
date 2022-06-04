import pathlib
from typing import Protocol

import numpy as np

from src.instance import Instance


class InstanceReader(Protocol):
    def read_instance(self, input_path: pathlib.Path) -> Instance:
        """Read instance data."""
        raise NotImplementedError()


class TypeAInstanceReader:
    def __init__(self, input_path: pathlib.Path) -> None:
        self.input_path = input_path

    def read_instance(self) -> Instance:
        """Read instance data from a text file."""
        def _str_to_nums(s: str) -> list[int]:
            return [int(num) for num in s.split()]

        with open(self.input_path, 'r') as fp:
            lines = fp.read().splitlines()

        num_items, capacity = _str_to_nums(lines[0])
        items = np.array([_str_to_nums(line) for line in lines[1:-1]])
        optimum = int(lines[-1])

        return Instance(**{'capacity': capacity,
                           'num_items': num_items,
                           'weights': items[:, 1],
                           'profits': items[:, 0],
                           'optimum': optimum,
                           'optimal_selection': np.full(num_items, -1)})


class TypeBInstanceReader:
    def __init__(self, input_path: pathlib.Path) -> None:
        self.input_path = input_path

    def read_instance(self) -> Instance:
        """Read instance data from a directory."""
        def _compute_optimum(profits: np.ndarray, optimal_selection: np.ndarray) -> int:
            return np.sum(profits * optimal_selection)

        capacity_file, weights_file, profits_file, solution_file = list(
            self.input_path.iterdir())

        with open(capacity_file, 'r') as fp:
            capacity = fp.read().strip()

        with open(weights_file, 'r') as fp:
            weights = fp.read().splitlines()

        with open(profits_file, 'r') as fp:
            profits = fp.read().splitlines()

        with open(solution_file, 'r') as fp:
            optimal_selection = fp.read().splitlines()

        capacity = int(capacity)
        weights = np.array([int(w) for w in weights])
        profits = np.array([int(p) for p in profits])
        num_items = len(weights)
        optimal_selection = np.array([int(x) for x in optimal_selection])
        optimum = _compute_optimum(profits, optimal_selection)

        return Instance(**{'capacity': capacity,
                           'num_items': num_items,
                           'weights': weights,
                           'profits': profits,
                           'optimum': optimum,
                           'optimal_selection': optimal_selection})
