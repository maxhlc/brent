# Standard imports
from glob import glob

# Orekit imports
import orekit
from org.orekit.data import DataSource
from org.orekit.files.sp3 import SP3Parser
from org.orekit.propagation.analytical import AggregateBoundedPropagator
import java.util

# Internal imports
import brent.propagators


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
