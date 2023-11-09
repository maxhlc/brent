# Standard imports
from datetime import datetime
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


def load_tle(path, start=datetime.min, end=datetime.max):
    # Read TLEs
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


def load_sp3(path, satID):
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
    sp3propagator = AggregateBoundedPropagator(sp3propagators)

    # Return aggregated propagator
    return sp3propagator
