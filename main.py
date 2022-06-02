import argparse
import pathlib

from instance import InstanceData
from instance_reader import TypeAInstanceReader, TypeBInstanceReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', help='instance path (type A: file, type B: directory)')
    args = parser.parse_args()

    # rng = np.random.default_rng()
    instance_path = pathlib.Path(args.path)
    reader = TypeAInstanceReader() if instance_path.is_file() else TypeBInstanceReader()
    instance_data = reader.read_data(instance_path)
    instance = InstanceData(**instance_data)
    print(instance_data)
    print()
    print(instance)


if __name__ == '__main__':
    main()
