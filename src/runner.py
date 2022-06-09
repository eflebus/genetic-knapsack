import math
import random

import numpy as np

from src.instance import Instance
from src.solution import Individual, Population


class Runner:
    def __init__(self, instance: Instance, population: Population, elite_fraction: float, p_mutation: float, p_mating: float, num_generations: int) -> None:
        self.instance = instance
        self.population = population
        self.num_elite_parents = math.ceil(
            self.population.size * elite_fraction)
        self.num_elite_offspring = self.population.size - self.num_elite_parents
        self.p_mutation = p_mutation
        self.p_mating = p_mating
        self.num_generations = num_generations
        self.max_fitness = []
        self.avg_fitness = []
        self.best_solutions = []

    @staticmethod
    def evaluate_fitness(population: Population, instance: Instance) -> np.ndarray:
        items_weights = instance.items_weights
        items_profits = instance.items_profits
        chromosomes = [
            individual.chromosome for individual in population.individuals]
        weights = np.array([np.sum(chromosome * items_weights)
                           for chromosome in chromosomes])
        fitness = np.array([np.sum(chromosome * items_profits)
                           for chromosome in chromosomes])
        fitness[weights > instance.capacity] = 0

        return fitness

    def select_mating_pool(self) -> Population:
        """Roulette wheel."""
        fitness = self.evaluate_fitness(
            self.population, self.instance)
        selection_probs = fitness / (np.sum(fitness) + 1e-8) + 1e-8
        mating_pool = random.choices(
            self.population.individuals, k=self.population.size, weights=selection_probs)

        return Population(mating_pool)

    @staticmethod
    def mating(population: Population, p_mating: float) -> Population:
        """Population recombination."""
        return Population.from_mating(population.individuals, p_mating)

    @staticmethod
    def mutation(population: Population, p_mutation: float) -> None:
        population.mutation(p_mutation)

    def survival_selection(self, offspring: Population) -> None:
        """Elitist selection"""
        parents_fitness = self.evaluate_fitness(self.population, self.instance)
        offspring_fitness = self.evaluate_fitness(offspring, self.instance)

        elite_parents_idxs = np.argpartition(
            parents_fitness, -self.num_elite_parents)[-self.num_elite_parents:]
        elite_offspring_idxs = np.argpartition(
            offspring_fitness, -self.num_elite_offspring)[-self.num_elite_offspring:]

        elite_parents = [self.population.get_individual(i)
                         for i in elite_parents_idxs]
        elite_offspring = [offspring.get_individual(i)
                           for i in elite_offspring_idxs]
        self.population = Population(elite_parents + elite_offspring)

    def update_statistics(self) -> None:
        fitness = self.evaluate_fitness(self.population, self.instance)
        max_idx = np.argmax(fitness)
        self.max_fitness.append(fitness[max_idx])
        self.avg_fitness.append(np.mean(fitness))
        self.best_solutions.append(
            self.population.get_individual(max_idx))

    def best_solution(self) -> tuple[Individual, int, int]:
        best_population = Population(self.best_solutions)
        fitness = self.evaluate_fitness(best_population, self.instance)
        max_idx = np.argmax(fitness)
        best_individual = best_population.get_individual(max_idx)
        weight = np.sum(best_individual.chromosome *
                        self.instance.items_weights)
        return best_population.get_individual(max_idx), fitness[max_idx], weight

    def evolution_step(self) -> None:
        # Parents selection
        mating_pool = self.select_mating_pool()

        # Parents recombination via crossover
        offspring = self.mating(mating_pool, self.p_mating)

        # Offspring mutation
        self.mutation(offspring, self.p_mutation)

        # Select new population via elitism
        self.survival_selection(offspring)

        # Update population's fitness statistics
        self.update_statistics()

    def run(self) -> None:
        # Initialize population statistics
        self.update_statistics()

        # Run evolution
        for _ in range(self.num_generations):
            self.evolution_step()
