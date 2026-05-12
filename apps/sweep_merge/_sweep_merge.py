# Standard imports
from argparse import ArgumentParser, Namespace
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
