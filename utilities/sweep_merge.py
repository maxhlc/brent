# Standard imports
from argparse import ArgumentParser
from typing import List

# Third-party imports
import pandas as pd


def main(input: List[str], output: str) -> None:
    # Load
    dfs = [pd.read_pickle(path) for path in input]

    # Merge
    # TODO: handle duplicate points
    df = pd.concat(dfs, ignore_index=True)

    # Save
    df.to_pickle(output)


if __name__ == "__main__":
    # Parse file paths
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", action="append", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser_args = parser.parse_args()

    # Execute main function
    main(parser_args.input, parser_args.output)
