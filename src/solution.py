import numpy as np


class Individual:
    # TODO: add random mutation method
    def __init__(self, chromosome: np.ndarray) -> None:
        self.chromosome = chromosome
        self.chromosome_size = len(chromosome)

    @classmethod
    def from_random_chromosome(cls, chromosome_size: int):
        return cls(np.random.randint(2, size=chromosome_size, dtype=np.uint8))

    def evaluate_attribute(self, items_property: np.ndarray) -> int:
        return np.sum(self.chromosome * items_property)


class Population:
    def __init__(self, individuals: list[Individual]) -> None:
        self.individuals = individuals
        self.population_size = len(individuals)
        self.chromosome_size = individuals[0].chromosome_size

    @classmethod
    def from_random_individuals(cls, population_size: int, chromosome_size: int):
        return cls([Individual.from_random_chromosome(chromosome_size) for _ in range(population_size)])

    def evaluate_attribute(self, items_property: np.ndarray) -> np.ndarray:
        return np.array([individual.evaluate_attribute(items_property) for individual in self.individuals])
