# Future imports
from __future__ import annotations

# Standard imports
from datetime import datetime
from typing import Dict, Any

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from org.orekit.propagation.analytical.tle import (
    TLE,
    TLEPropagator as OrekitTLEPropagator,
)

# Internal imports
from brent import Constants
from brent.bias import BiasModel
from brent.noise import CovarianceProvider
from .propagator import Propagator, WrappedPropagator


class TLEPropagator(Propagator):
    # Declare TLE path and start/end dates
    path: str = ""
    start: datetime = datetime.min
    end: datetime = datetime.max

    def __init__(
        self,
        tle: list[TLE],
        bias: BiasModel = BiasModel(),
        noise: CovarianceProvider = CovarianceProvider(),
    ):
        # TODO: allow input of single TLE

        # Check for number of TLEs
        if len(tle) < 1:
            raise ValueError("Insufficent number of TLEs")

        # Store epochs and propagators
        epochs = [absolutedate_to_datetime(itle.getDate()) for itle in tle]
        propagators = [
            WrappedPropagator(OrekitTLEPropagator.selectExtrapolator(itle), bias, noise)
            for itle in tle
        ]

        # Store sorted epoch/propagator pairs
        self.propagators = sorted(zip(epochs, propagators), key=lambda x: x[0])

        # Initialise parent
        super().__init__(bias, noise)

    def _propagate(self, date, frame=Constants.DEFAULT_ECI):
        # Select propagator
        epochs = np.array([propagator[0] for propagator in self.propagators])
        idx: int = np.searchsorted(epochs, date, side="left")

        # Clip index to prevent out-of-range list access
        # NOTE: use last TLE in set if none available ahead of propagation date
        idx = np.clip(idx, 0, len(epochs) - 1)

        # Extract corresponding propagator
        propagator = self.propagators[idx][1]

        # Return propagated state
        return propagator._propagate(date, frame)

    @staticmethod
    def _load(
        path: str,
        start: datetime = datetime.min,
        end: datetime = datetime.max,
    ) -> list[TLE]:
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

    @staticmethod
    def load(
        path: str,
        start: datetime = datetime.min,
        end: datetime = datetime.max,
    ) -> TLEPropagator:
        # Load TLEs
        tles = TLEPropagator._load(path, start, end)

        # Create propagator
        tlePropagator = TLEPropagator(tles)

        # Assign path and start/end dates
        tlePropagator.path = path
        tlePropagator.start = start
        tlePropagator.end = end

        # Return TLE propagator
        return tlePropagator

    def serialise(self) -> Dict[str, Any]:
        # TODO: dump TLEs to temporary file
        if self.path == "":
            raise ValueError("Unable to serialise TLEPropagator without path")

        # Return serialised propagator
        return {
            "type": "tle",
            "parameters": {
                "path": self.path,
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
            },
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> TLEPropagator:
        # Assert type matches
        assert struct["type"] == "tle"

        # Extract path and identifier
        path = str(struct["parameters"]["path"])
        start = datetime.fromisoformat(struct["parameters"]["start"])
        end = datetime.fromisoformat(struct["parameters"]["end"])

        # Return TLE propagator
        return TLEPropagator.load(path, start, end)
