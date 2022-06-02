import pathlib
from typing import Protocol

import numpy as np


class InstanceReader(Protocol):
    def read_data(self, instance_path: pathlib.Path) -> dict:
        """Read instance data."""
        raise NotImplementedError()


class TypeAInstanceReader:
    def read_data(self, instance_path: pathlib.Path) -> dict:
        """Read instance data from a single text file."""
        def _str_to_nums(s: str) -> list[int]:
            return [int(num) for num in s.split()]

        with open(instance_path, 'r') as fp:
            lines = fp.read().splitlines()

        num_items, capacity = _str_to_nums(lines[0])
        items = np.array([_str_to_nums(line) for line in lines[1:-1]])
        optimum = int(lines[-1])

        return {'capacity': capacity,
                'num_items': num_items,
                'items': items,
                'optimum': optimum,
                'optimal_selection': np.full(num_items, -1)}


class TypeBInstanceReader:
    def read_data(self, instance_path: pathlib.Path) -> dict:
        """Read instance data from a directory containing several text files."""
        def _compute_optimum(profits: np.ndarray, optimal_selection: np.ndarray) -> int:
            return np.sum(profits * optimal_selection)

        capacity_file, weights_file, profits_file, solution_file = list(
            instance_path.iterdir())

        with open(capacity_file, 'r') as fp:
            capacity = fp.read().strip()

        with open(weights_file, 'r') as fp:
            weights = fp.read().splitlines()

        with open(profits_file, 'r') as fp:
            profits = fp.read().splitlines()

        with open(solution_file, 'r') as fp:
            optimal_selection = fp.read().splitlines()

        capacity = int(capacity)
        items = np.array([[int(w), int(p)] for w, p in zip(weights, profits)])
        num_items = len(items)
        optimal_selection = np.array([int(x) for x in optimal_selection])
        optimum = _compute_optimum(items[:, 1], optimal_selection)

        return {'capacity': capacity,
                'num_items': num_items,
                'items': items,
                'optimum': optimum,
                'optimal_selection': optimal_selection}
