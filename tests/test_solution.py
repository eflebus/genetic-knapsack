import numpy as np
from src.solution import Individual, Population


def test_individual_init() -> None:
    individual = Individual(chromosome_size=8)
    assert np.array_equal(individual.chromosome, np.zeros(8))
    assert individual.chromosome.dtype == np.uint8
    assert individual.compute_fitness(items_profits=np.arange(1, 9)) == 0
    assert individual.compute_weight(items_weights=np.arange(10, 18)) == 0


def test_population_init() -> None:
    population = Population(population_size=5, chromosome_size=3)
    assert len(population.individuals) == 5
    assert isinstance(population.individuals[0], Individual)
    assert np.array_equal(population.individuals[0].chromosome, np.zeros(3))
