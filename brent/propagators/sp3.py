# Future imports
from __future__ import annotations

# Standard imports
from glob import glob
from typing import Dict, Any

# Orekit imports
import orekit
from orekit.pyhelpers import absolutedate_to_datetime
from org.orekit.data import DataSource
from org.orekit.files.sp3 import SP3, SP3Parser
from org.orekit.propagation.analytical import AggregateBoundedPropagator
import java.util

# Internal imports
from .propagator import WrappedPropagator
from brent.bias import BiasModel
from brent.noise import CovarianceProvider


class SP3Propagator(WrappedPropagator):
    # Declare path and identifier
    path: str
    id: str

    @staticmethod
    def load(
        path: str,
        id: str,
        bias: BiasModel = BiasModel(),
        noise: CovarianceProvider = CovarianceProvider(),
    ) -> SP3Propagator:
        # Create glob of paths
        paths = glob(path)

        # Create SP3 parser
        sp3parser = SP3Parser()

        # Generate list of SP3s
        sp3list = [sp3parser.parse(DataSource(path)) for path in paths]

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

        # Create propagator
        sp3propagator = SP3Propagator(sp3aggregated, bias, noise)

        # Assign path and ID
        sp3propagator.path = path
        sp3propagator.id = id

        # Return propagator
        return sp3propagator

    def serialise(self) -> Dict[str, Any]:
        # Return serialised propagator
        return {
            "type": "sp3",
            "parameters": {
                "path": self.path,
                "id": self.id,
            },
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> SP3Propagator:
        # Assert type matches
        assert struct["type"] == "sp3"

        # Extract path and identifier
        path = struct["parameters"]["path"]
        id = struct["parameters"]["id"]

        # Return SP3 propagator
        return SP3Propagator.load(path, id)
