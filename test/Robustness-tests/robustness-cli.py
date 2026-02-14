from pathlib import Path
import argparse

from src.robustness.processor import RobustnessProcessor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mod",
        type=str,
        default="all",
        help="tipo de modificación"
    )

    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    processor = RobustnessProcessor(PROJECT_ROOT)

    processor.build_dataset()

    if args.mod == "all":
        processor.run_all()
    else:
        processor.run_modification(args.mod)


if __name__ == "__main__":
    main()
