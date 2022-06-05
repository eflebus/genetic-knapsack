import argparse
import pathlib

from src.instance_reader import TypeAInstanceReader, TypeBInstanceReader
from src.runner import Runner
from src.solution import Population


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='instance path (type A: file, type B: directory)')
    parser.add_argument('-p', '--population-size', type=int,
                        default=4, help='Number of individuals (default: %(default)s)')
    parser.add_argument('-e', '--elite-fraction', type=float, default=0.2,
                        help='Fraction of individuals to include in the next generation (default: %(default)s)')
    parser.add_argument('-m', '--mutation-prob', type=float, default=0.1,
                        help='Mutation probability (default: %(default)s)')
    parser.add_argument('-M', '--mating-prob', type=float, default=0.9,
                        help='Mating probability (default: %(default)s)')
    parser.add_argument('-n', '--num-generations', type=int, default=500,
                        help='Number of generations (default: %(default)s)')
    args = parser.parse_args()

    # rng = np.random.default_rng()
    input_path = pathlib.Path(args.path)
    reader = TypeAInstanceReader(input_path) if input_path.is_file(
    ) else TypeBInstanceReader(input_path)
    population_size = args.population_size
    elite_fraction = args.elite_fraction
    p_mutation = args.mutation_prob
    p_mating = args.mating_prob
    num_generations = args.num_generations
    instance = reader.read_instance()
    print(instance)
    population = Population.from_random_individuals(
        population_size=population_size, chromosome_size=len(instance.items_weights))
    runner = Runner(instance, population, elite_fraction,
                    p_mutation, p_mating, num_generations)
    runner.run()
    print('\nRESULTS')
    best_solution = runner.best_solution()
    print(f"Best solution: {best_solution}")
    print(
        f"  Fitness: {best_solution.evaluate_attribute(instance.items_profits)}")
    print(
        f"  Weight: {best_solution.evaluate_attribute(instance.items_weights)}")


if __name__ == '__main__':
    main()
