import numpy as np
from src.solution import Individual, Population

np.random.seed(42)  # For reproducibility


def test_individual_fixed_chromosome() -> None:
    chromosome = np.array([1, 0, 1, 0], dtype=np.uint8)
    individual = Individual(chromosome=chromosome)

    assert isinstance(individual.chromosome, np.ndarray)
    assert np.array_equal(individual.chromosome, chromosome)
    assert individual.num_genes == 4
    assert individual is not Individual(chromosome=chromosome)
    assert individual == Individual(chromosome=chromosome)
    assert f"{individual}" == str(chromosome)


def test_individual_random_chromosome() -> None:
    random_individual = Individual.from_random_chromosome(num_genes=8)

    assert isinstance(random_individual, Individual)
    assert random_individual.chromosome.dtype == np.uint8
    assert np.array_equal(random_individual.chromosome,
                          np.array([0, 0, 1, 1, 1, 1, 0, 1]))


def test_individual_mutation() -> None:
    chromosome = np.array([1, 0, 1, 0], dtype=np.uint8)
    individual = Individual(chromosome=chromosome)

    individual.mutation(p_mutation=0.5)
    assert np.array_equal(individual.chromosome, np.array([1, 0, 1, 1]))

    individual.mutation(p_mutation=1)
    assert np.array_equal(individual.chromosome, np.array([0, 1, 0, 0]))

    individual.mutation(p_mutation=0)
    assert np.array_equal(individual.chromosome, np.array([0, 1, 0, 0]))


def test_population_fixed_individuals() -> None:
    chromosome1 = np.array([0, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    chromosome2 = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
    individual1 = Individual(chromosome=chromosome1)
    individual2 = Individual(chromosome=chromosome2)
    population = Population(individuals=[individual1, individual2])

    assert population.size == 2
    assert isinstance(population.individuals, list)
    assert len(population.individuals) == population.size
    assert isinstance(population.individuals[0], Individual)
    assert population.individuals[0].chromosome.dtype == np.uint8
    assert np.array_equal(population.individuals[0].chromosome, chromosome1)
    assert np.array_equal(population.individuals[1].chromosome, chromosome2)
    assert population.individuals[0] == population.get_individual(idx=0)
    assert population.individuals[0] is population.get_individual(idx=0)
    assert population.individuals[0] == individual1
    assert population.individuals[1] == individual2
    assert population.individuals[0] is not individual1
    assert population.individuals[1] is not individual2
    assert str(
        population) == f"Individual #0: {chromosome1}\nIndividual #1: {chromosome2}"


def test_population_random_individuals() -> None:
    random_population = Population.from_random_individuals(size=5, num_genes=6)

    assert isinstance(random_population, Population)
    assert isinstance(random_population.individuals, list)
    assert len(random_population.individuals) == 5


def test_population_mating() -> None:
    chromosome1 = np.array([0, 0, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
    chromosome2 = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
    individual1 = Individual(chromosome=chromosome1)
    individual2 = Individual(chromosome=chromosome2)

    parents = Population.from_mating(
        mating_pool=[individual1, individual2], p_mating=0)
    assert parents.size == 2
    assert parents.get_individual(idx=0) is not individual1
    assert parents.get_individual(idx=1) is not individual2
    assert parents.get_individual(idx=0) == individual1
    assert parents.get_individual(idx=1) == individual2

    offspring = Population.from_mating(
        mating_pool=[individual1, individual2], p_mating=1)
    assert offspring.size == 2
    assert offspring.get_individual(idx=0) is not individual1
    assert offspring.get_individual(idx=0) is not individual2
    assert offspring.get_individual(idx=0) != individual1
    assert offspring.get_individual(idx=1) != individual2
    assert np.array_equal(
        offspring.individuals[0].chromosome, np.array([0, 1, 0, 0, 1, 1, 1, 0]))
    assert np.array_equal(
        offspring.individuals[1].chromosome, np.array([1, 0, 1, 0, 1, 0, 0, 1]))


def test_population_mutation() -> None:
    individual1 = Individual(chromosome=np.array([0, 1, 0, 0, 1, 1, 1, 0]))
    individual2 = Individual(chromosome=np.array([1, 0, 1, 0, 1, 0, 0, 1]))
    population = Population(individuals=[individual1, individual2])
    population.mutation(p_mutation=0)
    assert np.array_equal(
        population.individuals[0].chromosome, np.array([0, 1, 0, 0, 1, 1, 1, 0]))
    assert np.array_equal(
        population.individuals[1].chromosome, np.array([1, 0, 1, 0, 1, 0, 0, 1]))

    population.mutation(p_mutation=1)
    assert np.array_equal(
        population.individuals[0].chromosome, np.array([1, 0, 1, 1, 0, 0, 0, 1]))
    assert np.array_equal(
        population.individuals[1].chromosome, np.array([0, 1, 0, 1, 0, 1, 1, 0]))
