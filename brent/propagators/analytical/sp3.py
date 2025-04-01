# Future imports
from __future__ import annotations

# Standard imports
from glob import glob
from functools import cache

# Orekit imports
import orekit
from orekit.pyhelpers import absolutedate_to_datetime
from org.orekit.data import DataSource
from org.orekit.files.sp3 import SP3, SP3Parser
from org.orekit.propagation.analytical import AggregateBoundedPropagator
import java.util

# Internal imports
from brent.propagators import WrappedPropagator


class SP3Propagator(WrappedPropagator):

    # TODO: replace caching with propagator splitting by id in load method?
    @cache
    @staticmethod
    def _parse(path: str) -> SP3:
        # Return parsed SP3 file
        return SP3Parser().parse(DataSource(path))

    @staticmethod
    def load(path: str, id: str) -> SP3Propagator:
        # Create glob of paths
        paths = glob(path)

        # Generate list of SP3s
        sp3list: list[SP3] = []

        # Iterate through paths
        for path in paths:
            try:
                # Load SP3 file
                sp3 = SP3Propagator._parse(path)

                # Append to list of SP3s
                sp3list.append(sp3)
            except orekit.JavaError as e:
                # TODO: log error?
                pass

        # Declare list for SP3 propagators
        sp3propagators = []

        # Iterate through SP3 data
        for sp3 in sp3list:
            # Extract segments
            segments = sp3.getEphemeris(id).getSegments()

            # Extract number of segments
            ns = segments.size()

            # Iterate through segments
            for idx in range(ns):
                # Extract propagator
                propagator = segments.get(idx).getPropagator()

                # Add to propagator list
                sp3propagators.append(propagator)

        # Sort propagators by ascending starting date
        sp3propagators.sort(key=lambda x: absolutedate_to_datetime(x.getMinDate()))

        # Declare Java list for SP3 propagators
        sp3propagators_ = java.util.ArrayList()
        [sp3propagators_.add(sp3propagator) for sp3propagator in sp3propagators]

        # Create aggregate propagator
        sp3aggregated = AggregateBoundedPropagator(sp3propagators_)

        # Return propagator
        return SP3Propagator(sp3aggregated)
