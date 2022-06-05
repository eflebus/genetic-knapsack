import math
import random

import numpy as np

from src.instance import Instance
from src.solution import Population


class Runner:
    def __init__(self, instance: Instance, population: Population, elite_fraction: float, p_mutation: float, p_mating: float) -> None:
        self.instance = instance
        self.population = population
        self.elite_size = math.ceil(
            self.population.population_size * elite_fraction)
        self.num_offspring = self.population.population_size - self.elite_size
        self.p_mutation = p_mutation
        self.p_mating = p_mating

    @staticmethod
    def evaluate_fitness(population: Population, instance: Instance) -> np.ndarray:
        weights = population.evaluate_attribute(instance.items_weights)
        fitness = population.evaluate_attribute(instance.items_profits)
        fitness[weights > instance.capacity] = 0
        return fitness

    def select_mate_pool(self) -> Population:
        """Roulette wheel."""
        individuals_fitness = self.evaluate_fitness(
            self.population, self.instance)
        selection_probs = individuals_fitness / np.sum(individuals_fitness)
        print(f"Selection probs: {selection_probs}")
        selected_mates = random.choices(
            self.population.individuals, k=self.population.population_size, weights=selection_probs)
        # selection_probs = np.cumsum(individuals_fitness)
        # print(
        #     f"Selection probs: {selection_probs/np.sum(selection_probs)}")
        # selected_mates = random.choices(
        #     self.population.individuals, k=self.population.population_size, cum_weights=selection_probs)
        return Population(selected_mates)

    @staticmethod
    def mating(population: Population, p_mating: float) -> Population:
        """Population recombination (mating)."""
        return Population.from_mating(population.individuals, p_mating)

    @staticmethod
    def mutation(population: Population, p_mutation: float) -> None:
        population.mutate(p_mutation)

    def elitist_selection(self, offspring: Population) -> None:
        """Survivals, elitism."""
        print('PARENTS')
        print(self.population)
        print('OFFSPRING')
        print(offspring)
        parents_fitness = self.evaluate_fitness(self.population, self.instance)
        offspring_fitness = self.evaluate_fitness(offspring, self.instance)
        elite_parents_idxs = np.argpartition(
            parents_fitness, -self.elite_size)[-self.elite_size:]
        elite_offspring_idxs = np.argpartition(
            offspring_fitness, -self.num_offspring)[-self.num_offspring:]
        print(f"elite_parents_idxs: {elite_parents_idxs}")
        print(f"elite_offspring_idxs: {elite_offspring_idxs}")
        elite_parents = [self.population.individuals[i]
                         for i in elite_parents_idxs]
        elite_offspring = [offspring.individuals[i]
                           for i in elite_offspring_idxs]
        self.population = Population(elite_parents + elite_offspring)
