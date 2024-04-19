# Future imports
from __future__ import annotations

# Standard imports
from glob import glob

# Orekit imports
import orekit
from org.orekit.data import DataSource
from org.orekit.files.sp3 import SP3, SP3Parser
from org.orekit.propagation.analytical import AggregateBoundedPropagator
import java.util

# Internal imports
from .propagator import WrappedPropagator


class SP3Propagator(WrappedPropagator):

    @staticmethod
    def load(path: str, id: str) -> SP3Propagator:
        # Create glob of paths
        paths = glob(path)

        # Create SP3 parser
        sp3parser = SP3Parser()

        # Declare list for SP3 files
        sp3list = java.util.ArrayList()

        # Iterate through paths
        for path in paths:
            # Open file
            sp3file = DataSource(path)

            # Parse SP3 data
            sp3data = sp3parser.parse(sp3file)

            # Add SP3 data to list
            sp3list.add(sp3data)

        # Splice SP3 files together
        sp3 = SP3.splice(sp3list)

        # TODO: fix SP3Ephemeris.getPropagator() not available through Orekit wrapper

        # Declare list for SP3 propagators
        sp3propagators = java.util.ArrayList()

        # Extract SP3 segments
        segments = sp3.getEphemeris(id).getSegments()

        # Extract number of segments
        nsegments = segments.size()

        # Iterate through segments
        for idx in range(nsegments):
            # Extract propagator
            sp3propagators.add(segments.get(idx).getPropagator())

        # Create aggregate propagator
        sp3aggregated = AggregateBoundedPropagator(sp3propagators)

        # Return propagator
        return SP3Propagator(sp3aggregated)
