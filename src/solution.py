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

    def split_chromosome(self, split_idx: int) -> list[np.ndarray]:
        return np.split(self.chromosome, [split_idx])

    def __str__(self) -> str:
        return f"{self.chromosome}"


class Population:
    def __init__(self, individuals: list[Individual]) -> None:
        self.individuals = individuals
        self.population_size = len(individuals)

    @classmethod
    def from_random_individuals(cls, population_size: int, chromosome_size: int):
        return cls([Individual.from_random_chromosome(chromosome_size) for _ in range(population_size)])

    @classmethod
    def from_mating(cls, individuals: list[Individual], p_mating: float):
        offspring = []
        num_individuals = len(individuals)

        for parent1, parent2 in zip(individuals, individuals[1:]):
            new_individual1, new_individual2 = parent1, parent2

            if np.random.random() <= p_mating:
                cross_idx = np.random.randint(1, num_individuals - 1)
                new_individual1, new_individual2 = cls.single_point_crossover(
                    new_individual1, new_individual2, cross_idx)

            offspring.extend((new_individual1, new_individual2))

        return cls(offspring)

    def evaluate_attribute(self, items_property: np.ndarray) -> np.ndarray:
        return np.array([individual.evaluate_attribute(items_property) for individual in self.individuals])

    def mutate(self, p_mutation: float) -> None:
        for individual in self.individuals:
            individual.mutate(p_mutation)

    @staticmethod
    def single_point_crossover(parent1: Individual, parent2: Individual, cross_idx: int) -> tuple[Individual, Individual]:
        """Single-point crossover recombination."""
        chromo1_part1, chromo1_part2 = parent1.split_chromosome(cross_idx)
        chromo2_part1, chromo2_part2 = parent2.split_chromosome(cross_idx)

        new_chromoA = np.concatenate((chromo1_part1, chromo2_part2))
        new_chromoB = np.concatenate((chromo1_part2, chromo2_part1))

        return Individual(new_chromoA), Individual(new_chromoB)

    def fittest_individual(self, items_profit: np.ndarray) -> Individual:
        return self.individuals[np.argmax(self.evaluate_attribute(items_profit))]

    def __str__(self) -> str:
        return '\n'.join([f"Individual #{i}: {individual}" for i, individual in enumerate(self.individuals, start=1)])
