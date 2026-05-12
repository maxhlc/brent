# Future imports
from __future__ import annotations

# Standard imports
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Internal imports
from brent.propagators import NumericalPropagatorParameters


@dataclass
class Arguments:
    # Fit window parameters
    start: datetime
    duration: timedelta

    # TLE parameters
    tle: str

    # SP3 parameters
    sp3: str
    sp3name: str

    # Model parameters
    model: NumericalPropagatorParameters

    # Output parameters
    verbose: bool
    plot: bool
    output: str

    @staticmethod
    def load(path: str) -> Arguments:
        # Load argument file
        with open(path, "r") as fid:
            args = json.load(fid)

        # Cast required types
        args["start"] = datetime.strptime(args["start"], "%Y-%m-%d")
        args["duration"] = timedelta(args["duration"])
        args["model"] = NumericalPropagatorParameters(**args["model"])

        # Return arguments
        return Arguments(**args)
