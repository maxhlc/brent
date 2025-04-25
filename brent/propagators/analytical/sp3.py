# Future imports
from __future__ import annotations

# Standard imports
from glob import glob
from functools import cache

# Orekit imports
import orekit
from orekit.pyhelpers import absolutedate_to_datetime
from org.orekit.data import DataSource
from org.orekit.files.sp3 import SP3, SP3Parser, SP3Segment
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

    @classmethod
    def _generate(cls, sp3list: list[SP3], id: str) -> SP3Propagator:
        # Declare list for SP3 propagators
        sp3propagators = []

        # Iterate through SP3 data
        for sp3 in sp3list:
            try:
                # Extract segments
                segments: list[SP3Segment] = list(sp3.getEphemeris(id).getSegments())

                # Iterate through segments
                for segment in segments:
                    # Extract propagator
                    propagator = segment.getPropagator()

                    # Add to propagator list
                    sp3propagators.append(propagator)

            except orekit.JavaError:
                # TODO: log error?
                continue

        # Sort propagators by ascending starting date
        sp3propagators.sort(key=lambda x: absolutedate_to_datetime(x.getMinDate()))

        # Declare Java list for SP3 propagators
        sp3propagators_ = java.util.ArrayList()
        [sp3propagators_.add(sp3propagator) for sp3propagator in sp3propagators]

        # Create aggregate propagator
        sp3aggregated = AggregateBoundedPropagator(sp3propagators_)

        # Return propagator
        return SP3Propagator(sp3aggregated)

    @classmethod
    def load(cls, path: str, id: str) -> SP3Propagator:
        # Create glob of paths
        paths = glob(path, recursive=True)

        # Generate list of SP3s
        sp3list: list[SP3] = []

        # Iterate through paths
        for path in paths:
            try:
                # Load SP3 file
                sp3 = SP3Propagator._parse(path)

                # Append to list of SP3s
                sp3list.append(sp3)
            except orekit.JavaError:
                # TODO: log error?
                pass

        # Return propagator
        return cls._generate(sp3list, id)

    @classmethod
    def load_all(cls, path: str) -> dict[str, SP3Propagator]:
        # Create glob of paths
        paths = glob(path, recursive=True)

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

        # Extract SP3 identifiers
        sp3headers = [sp3.getHeader() for sp3 in sp3list]
        sp3ids = set(
            [id for sp3header in sp3headers for id in list(sp3header.getSatIds())]
        )

        # Generate propagators
        propagators = {id: cls._generate(sp3list, id) for id in sp3ids}

        # Return propagators
        return propagators
