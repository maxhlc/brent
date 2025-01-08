# Standard imports
from argparse import ArgumentParser, Namespace
from typing import List

# Third-party imports
import pandas as pd

# Internal imports
from .application import Application, ApplicationFactory


def main(input: List[str], output: str) -> None:
    # Load
    dfs = [pd.read_pickle(path) for path in input]

    # Merge
    # TODO: handle duplicate points
    df = pd.concat(dfs, ignore_index=True)

    # Save
    df.to_pickle(output)


@ApplicationFactory.register("sweep_merge", "Merge results from multiple sweeps")
class SweepMerge(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Extract arguments
        input = arguments.input
        output = arguments.output

        # Execute sweep merge
        main(input, output)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument(
            "-i",
            "--input",
            action="append",
            type=str,
            required=True,
            help="Input filepath(s)",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="Output filepath",
        )
