from dataclasses import dataclass, field

import numpy as np


@dataclass
class Individual:
    chromosome_size: int
    chromosome: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.chromosome = np.random.randint(
            2, size=self.chromosome_size, dtype=np.uint8)

    def compute_fitness(self, items_profits: np.ndarray) -> int:
        return np.sum(self.chromosome * items_profits)

    def compute_weight(self, items_weights: np.ndarray) -> int:
        return np.sum(self.chromosome * items_weights)


@dataclass
class Population:
    population_size: int
    chromosome_size: int
    individuals: list[Individual] = field(init=False)

    def __post_init__(self) -> None:
        self.individuals = [Individual(self.chromosome_size)
                            for _ in range(self.population_size)]

    def compute_fitness(self, items_profits: np.ndarray) -> np.ndarray:
        return np.array([individual.compute_fitness(items_profits) for individual in self.individuals])

    def compute_weight(self, items_weights: np.ndarray) -> np.ndarray:
        return np.array([individual.compute_weight(items_weights) for individual in self.individuals])
