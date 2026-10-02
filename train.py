"""Top-level launcher for the NASA, Ahmed, Poisson, and Darcy benchmarks."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


EXPERIMENT_MODULES = {
    "nasa": "nasa.train",
    "ahmed": "ahmed.train",
    "poisson": "poisson.train",
    "darcy": "darcy.train",
}


def _load_experiment(name: str):
    """Load a benchmark module from this project, not a namesake elsewhere."""

    if __package__:
        return importlib.import_module(f"{__package__}.{EXPERIMENT_MODULES[name]}")

    project_directory = Path(__file__).resolve().parent
    project_path = str(project_directory)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)
    return importlib.import_module(EXPERIMENT_MODULES[name])


def main(argv: list[str] | None = None) -> None:
    """Parse the benchmark name and forward all remaining arguments."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        choices=tuple(EXPERIMENT_MODULES),
        help="Benchmark to reproduce.",
    )
    if not arguments or arguments in (["-h"], ["--help"]):
        parser.print_help()
        return

    # Parse only the first positional argument so ``train.py nasa --help``
    # delegates ``--help`` to the NASA argument parser.
    args = parser.parse_args(arguments[:1])
    remaining = arguments[1:]
    module = _load_experiment(args.experiment)
    module.main(remaining)


if __name__ == "__main__":
    main()
