import numpy as np


class Individual:
    def __init__(self, chromosome: np.ndarray) -> None:
        self._chromosome = chromosome
        self.num_genes = len(chromosome)

    @classmethod
    def from_random_chromosome(cls, num_genes: int):
        return cls(np.random.randint(2, size=num_genes, dtype=np.uint8))

    @property
    def chromosome(self) -> np.ndarray:
        return self._chromosome

    def mutation(self, p_mutation: float) -> None:
        mutation_idxs = np.random.random(
            size=self.num_genes) <= p_mutation
        self._chromosome[mutation_idxs] = 1 - self._chromosome[mutation_idxs]

    def split_chromosome(self, split_idx: int) -> list[np.ndarray]:
        return np.split(self._chromosome, [split_idx])

    def __eq__(self, other):
        return np.array_equal(self._chromosome, other._chromosome)

    def __str__(self) -> str:
        return f"{self._chromosome}"


class Population:
    def __init__(self, individuals: list[Individual]) -> None:
        self._individuals = individuals
        self.size = len(individuals)

    @classmethod
    def from_random_individuals(cls, size: int, num_genes: int):
        return cls([Individual.from_random_chromosome(num_genes) for _ in range(size)])

    @classmethod
    def from_mating(cls, mating_pool: list[Individual], p_mating: float):
        offspring = []
        num_individuals = len(mating_pool)

        for parent1, parent2 in zip(mating_pool, mating_pool[1:]):
            new_individual1, new_individual2 = parent1, parent2

            if np.random.random() <= p_mating:
                cross_idx = np.random.randint(1, num_individuals)
                new_individual1, new_individual2 = cls.single_point_crossover(
                    new_individual1, new_individual2, cross_idx)

            offspring.extend((new_individual1, new_individual2))

        return cls(offspring)

    @property
    def individuals(self) -> list[Individual]:
        return self._individuals

    def get_individual(self, idx: int) -> Individual:
        return self._individuals[idx]

    def mutation(self, p_mutation: float) -> None:
        for individual in self._individuals:
            individual.mutation(p_mutation)

    @staticmethod
    def single_point_crossover(parent1: Individual, parent2: Individual, cross_idx: int) -> tuple[Individual, Individual]:
        chromo1_part1, chromo1_part2 = parent1.split_chromosome(cross_idx)
        chromo2_part1, chromo2_part2 = parent2.split_chromosome(cross_idx)

        new_chromo_A = np.concatenate((chromo1_part1, chromo2_part2))
        new_chromo_B = np.concatenate((chromo2_part1, chromo1_part2))

        return Individual(new_chromo_A), Individual(new_chromo_B)

    def __str__(self) -> str:
        return '\n'.join([f"Individual #{i}: {individual}" for i, individual in enumerate(self._individuals)])
