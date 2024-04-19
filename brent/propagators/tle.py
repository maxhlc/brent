# Future imports
from __future__ import annotations

# Standard imports
from datetime import datetime

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from org.orekit.orbits import CartesianOrbit, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical.tle import (
    TLE,
    TLEPropagator as OrekitTLEPropagator,
)
from org.orekit.propagation.analytical.tle.generation import (
    FixedPointTleGenerationAlgorithm,
)
from org.orekit.propagation.conversion import TLEPropagatorBuilder
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from brent import Constants
from .propagator import Propagator, WrappedPropagator


class TLEPropagator(Propagator):

    def __init__(self, tle: list[TLE]):
        # TODO: allow input of single TLE

        # Check for number of TLEs
        if len(tle) < 1:
            raise ValueError("Insufficent number of TLEs")

        # Store epochs and propagators
        epochs = [absolutedate_to_datetime(itle.getDate()) for itle in tle]
        propagators = [
            WrappedPropagator(OrekitTLEPropagator.selectExtrapolator(itle))
            for itle in tle
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
    ) -> TLEPropagator:
        # Load TLEs
        tles = TLEPropagator.__load(path, start, end)

        # Return TLE propagator
        return TLEPropagator(tles)

    @staticmethod
    def __template(date):
        # Default metadata
        satelliteNumber = 0
        classification = " "
        launchYear = 0
        launchNumber = 0
        launchPiece = "   "
        ephemerisType = 0
        elementNumber = 0
        revolutionNumberAtEpoch = 0

        # Set epoch
        epoch = datetime_to_absolutedate(date)

        # Default parameters
        meanMotion = 0.0
        meanMotionFirstDerivative = 0.0
        meanMotionSecondDerivative = 0.0
        e = 0.0
        i = 0.0
        pa = 0.0
        raan = 0.0
        meanAnomaly = 0.0
        bStar = 0.0

        # Return TLE template
        return TLE(
            satelliteNumber,
            classification,
            launchYear,
            launchNumber,
            launchPiece,
            ephemerisType,
            elementNumber,
            epoch,
            meanMotion,
            meanMotionFirstDerivative,
            meanMotionSecondDerivative,
            e,
            i,
            pa,
            raan,
            meanAnomaly,
            revolutionNumberAtEpoch,
            bStar,
        )

    @staticmethod
    def builder(date, state, bstar=False):
        # Convert date and state to Orekit format
        date_ = datetime_to_absolutedate(date)
        pos_ = Vector3D(*state[0:3].tolist())
        vel_ = Vector3D(*state[3:6].tolist())
        state_ = TimeStampedPVCoordinates(date_, pos_, vel_)
        orbit = CartesianOrbit(state_, Constants.DEFAULT_ECI, Constants.DEFAULT_MU)

        # Create template TLE
        templateTLE = TLEPropagator.__template(date)

        # Generate initial TLE
        tle = TLE.stateToTLE(
            SpacecraftState(orbit), templateTLE, FixedPointTleGenerationAlgorithm()
        )

        # Iterate through drivers
        for driver in tle.getParametersDrivers():
            # Enable BSTAR driver
            if driver.getName() == TLE.B_STAR:
                driver.setSelected(bstar)

        # Return TLE builder
        # TODO: change position scale?
        return TLEPropagatorBuilder(
            tle, PositionAngleType.MEAN, 1.0, FixedPointTleGenerationAlgorithm()
        )
