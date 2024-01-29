# Standard imports
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from glob import glob

# Third-party imports
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.data import DataSource
from org.orekit.files.sp3 import SP3Parser
from org.orekit.propagation.analytical import AggregateBoundedPropagator
from org.orekit.propagation.analytical.tle import TLE
import java.util

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


def load_tle(path, start=datetime.min, end=datetime.max):
    # Read TLEs
    # TODO: read glob like SP3 loader
    tles = pd.read_json(path)

    # Throw error if file contains multiple different objects
    if tles["OBJECT_ID"].nunique() != 1:
        raise ValueError("Multiple object identifiers in TLE file")

    # Sort by epoch and creation date
    tles.sort_values(["EPOCH", "CREATION_DATE"], inplace=True)

    # Drop duplicate TLEs, keeping the most recently issued instances
    tles.drop_duplicates("EPOCH", keep="last", inplace=True)

    # Generate TLEs
    tles = [
        TLE(tle["TLE_LINE1"], tle["TLE_LINE2"])
        for tle in tles.to_dict(orient="records")
    ]

    # Extract TLE subset
    tles = [
        tle
        for tle in tles
        if (tle.getDate().durationFrom(datetime_to_absolutedate(start)) >= 0)
        and (tle.getDate().durationFrom(datetime_to_absolutedate(end)) <= 0)
    ]

    # Return filtered TLEs
    return tles


def load_tle_propagator(path, start=datetime.min, end=datetime.max):
    # Load TLEs
    tles = load_tle(path, start, end)

    # Create propagator
    tlePropagator = brent.propagators.tles_to_propagator(tles)

    # Return propagator
    return brent.propagators.Propagator(tlePropagator)


def load_sp3_propagator(path, satID):
    # Create glob of paths
    paths = glob(path)

    # Create SP3 parser
    sp3parser = SP3Parser()

    # Declare list for propagators
    sp3propagators = java.util.ArrayList()

    # Iterate through paths
    for path in paths:
        # Open file
        sp3file = DataSource(path)

        # Parse SP3 data
        sp3data = sp3parser.parse(sp3file)

        # Extract propagator and add to list
        sp3propagators.add(sp3data.getSatellites().get(satID).getPropagator())

    # Aggregate propagators
    # TODO: replace with spliced SP3 files?
    sp3propagator = AggregateBoundedPropagator(sp3propagators)

    # Return aggregated propagator
    return brent.propagators.Propagator(sp3propagator)
