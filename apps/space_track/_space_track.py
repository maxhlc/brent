# Future imports
from __future__ import annotations

# Standard imports
from dataclasses import dataclass
from datetime import datetime
import json
from typing import List

# External imports
from brent.download import SpaceTrackDownloader


@dataclass
class Parameters:
    # Credentials
    username: str
    password: str

    # Paths
    output_directory: str

    # Identifiers
    identifiers: List[str]

    # Dates
    start: datetime
    end: datetime

    @staticmethod
    def load(fname: str) -> Parameters:
        # Load parameters from file
        with open(fname, "r") as fp:
            parameters = json.load(fp)

        # Format identifiers
        parameters["identifiers"] = [
            str(identifier) for identifier in parameters["identifiers"]
        ]

        # Format dates
        dateformat = "%Y-%m-%d"
        parameters["start"] = datetime.strptime(parameters["start"], dateformat)
        parameters["end"] = datetime.strptime(parameters["end"], dateformat)

        # Return parameters object
        return Parameters(**parameters)


def main(input: str) -> None:
    # Load parameters
    parameters = Parameters.load(input)

    # Create downloader
    downloader = SpaceTrackDownloader(
        parameters.username,
        parameters.password,
    )

    # Download element sets
    downloader.download(
        parameters.identifiers,
        parameters.start,
        parameters.end,
        parameters.output_directory,
    )
