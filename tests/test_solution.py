import numpy as np
from src.solution import Individual, Population

np.random.seed(42)


def test_individual() -> None:
    chromosome = np.array([1, 0, 1, 0], dtype=np.uint8)
    individual = Individual(chromosome=chromosome)
    random_individual = Individual.from_random_chromosome(chromosome_size=8)

    assert np.array_equal(individual.chromosome, chromosome)
    assert np.array_equal(random_individual.chromosome,
                          np.array([0, 0, 1, 1, 1, 1, 0, 1]))

    assert individual.chromosome_size == 4
    assert isinstance(random_individual, Individual)
    assert random_individual.chromosome.dtype == np.uint8

    assert individual.evaluate_attribute(
        items_property=np.array([5, 10, 0, 1])) == 5

    individual.mutate(p_mutation=0.5)
    assert np.array_equal(individual.chromosome, np.array([1, 0, 1, 1]))
    individual.mutate(p_mutation=1)
    assert np.array_equal(individual.chromosome, np.array([0, 1, 0, 0]))

    part1, part2 = random_individual.split_chromosome(split_idx=3)
    assert np.array_equal(part1, np.array([0, 0, 1]))
    assert np.array_equal(part2, np.array([1, 1, 1, 0, 1]))


def test_population() -> None:
    chromosome1 = np.array([1, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    chromosome2 = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
    individual1 = Individual(chromosome=chromosome1)
    individual2 = Individual(chromosome=chromosome2)
    population = Population(individuals=[individual1, individual2])

    assert population.population_size == 2
    assert isinstance(population.individuals[0], Individual)
    assert np.array_equal(population.individuals[0].chromosome, chromosome1)
    assert np.array_equal(population.individuals[1].chromosome, chromosome2)
