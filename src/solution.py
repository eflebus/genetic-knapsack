import numpy as np


class Individual:
    def __init__(self, chromosome: np.ndarray) -> None:
        self.chromosome = chromosome
        self.chromosome_size = len(chromosome)

    @classmethod
    def from_random_chromosome(cls, chromosome_size: int):
        return cls(np.random.randint(2, size=chromosome_size, dtype=np.uint8))

    def evaluate_attribute(self, items_property: np.ndarray) -> int:
        return np.sum(self.chromosome * items_property)

    def mutate(self, p_mutation: float) -> None:
        mutation_idxs = np.random.random(
            size=self.chromosome_size) <= p_mutation
        self.chromosome[mutation_idxs] = 1 - self.chromosome[mutation_idxs]

    def __str__(self) -> str:
        return f"{self.chromosome}"


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

    def mutate(self, p_mutation: int) -> None:
        for individual in self.individuals:
            individual.mutate(p_mutation)

    def __str__(self) -> str:
        return '\n'.join([f"Individual #{i}: {individual}" for i, individual in enumerate(self.individuals, start=1)])
