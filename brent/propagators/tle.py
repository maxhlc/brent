# Standard imports
from datetime import datetime

# Third-party imports
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.propagation.analytical import AggregateBoundedPropagator
from org.orekit.propagation.analytical.tle import (
    TLE,
    TLEPropagator as OrekitTLEPropagator,
)
import java.util

# Internal imports
from .propagator import Propagator


class TLEPropagator:

    @staticmethod
    def propagator(tles: list[TLE]) -> Propagator:
        # Check for number of TLEs
        if len(tles) < 1:
            raise ValueError("Insufficent number of TLEs")

        # TODO: sort TLEs to ensure in order?

        # Declare map for propagators
        propagatorMap = java.util.TreeMap()

        # Iterate through TLEs
        for tle in tles:
            # Extract epoch date
            epoch = tle.getDate()

            # Create propagator
            propagator = OrekitTLEPropagator.selectExtrapolator(tle)

            # Add to map
            propagatorMap.put(epoch, propagator)

        # Extract start and end dates
        # TODO: review dates
        dateStart = tles[0].getDate()
        dateEnd = tles[-1].getDate()

        # Return aggregate propagator
        tlePropagator = AggregateBoundedPropagator(propagatorMap, dateStart, dateEnd)

        # Return TLE propagator
        return Propagator(tlePropagator)

    @staticmethod
    def __load(
        path: str, start: datetime = datetime.min, end: datetime = datetime.min
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
        path: str, start: datetime = datetime.min, end: datetime = datetime.max
    ) -> Propagator:
        # Load TLEs
        tles = TLEPropagator.__load(path, start, end)

        # Return propagator
        return TLEPropagator.propagator(tles)
