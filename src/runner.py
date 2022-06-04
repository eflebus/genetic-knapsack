import random

import numpy as np

from src.instance import Instance
from src.solution import Population


class Runner:
    def __init__(self, instance: Instance, population: Population) -> None:
        self.instance = instance
        self.population = population

    def evaluate_fitness(self) -> np.ndarray:
        weights = self.population.evaluate_attribute(
            self.instance.items_weights)
        fitness = self.population.evaluate_attribute(
            self.instance.items_profits)
        fitness[weights > self.instance.capacity] = 0
        return fitness

    def select_mate_pool(self) -> Population:
        individuals_fitness = self.evaluate_fitness()
        selection_probs = individuals_fitness / np.sum(individuals_fitness)
        print(f"Selection probs: {selection_probs}")
        selected_mates = random.choices(
            self.population.individuals, k=self.population.population_size, weights=selection_probs)
        return Population(selected_mates)
