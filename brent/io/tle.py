# Standard imports
from datetime import datetime

# Third-party imports
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.propagation.analytical.tle import TLE

# Internal imports
import brent.propagators


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
