# Future imports
from __future__ import annotations

# Standard imports
from datetime import datetime
from glob import glob
from functools import cache

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
from brent.propagators import Propagator, WrappedPropagator


class TLEPropagator(Propagator):

    def __init__(self, tle: TLE | list[TLE]):
        # Ensure list of TLEs
        tles = tle if not isinstance(tle, TLE) else [tle]

        # Check for number of TLEs
        if len(tles) < 1:
            raise ValueError("Insufficent number of TLEs")

        # Store epochs and propagators
        epochs = [absolutedate_to_datetime(itle.getDate()) for itle in tles]
        propagators = [
            WrappedPropagator(OrekitTLEPropagator.selectExtrapolator(itle))
            for itle in tles
        ]

        # Store sorted epoch/propagator pairs
        self.propagators = sorted(zip(epochs, propagators), key=lambda x: x[0])

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

    @cache
    @staticmethod
    def _parse(path: str) -> pd.DataFrame:
        # Return parsed TLE file
        return pd.read_json(path)

    @classmethod
    def _load(
        cls,
        path: str,
        start: datetime = datetime.min,
        end: datetime = datetime.max,
        norad: int | None = None,
        cospar: str | None = None,
    ) -> list[TLE]:
        # Read TLEs
        paths = sorted(glob(path, recursive=True))
        tles = pd.concat([cls._parse(path) for path in paths], ignore_index=True)

        # Filter TLEs
        if norad is not None:
            tles = tles[tles["NORAD_CAT_ID"] == norad]
        if cospar is not None:
            tles = tles[tles["OBJECT_ID"] == cospar]

        # Throw error if file contains zero or multiple different objects
        nunique = tles["OBJECT_ID"].nunique()
        if nunique != 1:
            raise ValueError(f"{nunique} object identifiers")

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

    @classmethod
    def load(
        cls,
        path: str,
        start: datetime = datetime.min,
        end: datetime = datetime.max,
        norad: int | None = None,
        cospar: str | None = None,
    ) -> TLEPropagator:
        # Load TLEs
        tles = cls._load(path, start, end, norad, cospar)

        # Return TLE propagator
        return TLEPropagator(tles)
