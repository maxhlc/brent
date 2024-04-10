# Standard imports
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Internal imports
import brent.propagators


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
    model: brent.propagators.ModelParameters

    # Output parameters
    verbose: bool
    plot: bool
    output: str


def load_arguments(path):
    # Load argument file
    with open(path, "r") as fid:
        args = json.load(fid)

    # Cast required types
    args["start"] = datetime.strptime(args["start"], "%Y-%m-%d")
    args["duration"] = timedelta(args["duration"])
    args["model"] = brent.propagators.ModelParameters(**args["model"])

    # Return arguments
    return Arguments(**args)
