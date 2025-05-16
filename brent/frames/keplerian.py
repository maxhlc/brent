# Standard imports
from datetime import datetime
from typing import List

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.frames import Frame
from org.orekit.orbits import KeplerianOrbit
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from .angle import AngleType
from brent import Constants


class Keplerian:

    @staticmethod
    def from_cartesian(
        dates: List[datetime] | pd.DatetimeIndex,
        states: np.ndarray,
        angle: AngleType = AngleType.MEAN,
        mu: float = Constants.DEFAULT_MU,
        frame: Frame = Constants.DEFAULT_ECI,
    ) -> np.ndarray:
        # Return Keplerian elements
        return np.array(
            [
                Keplerian._from_cartesian(date, state, angle, mu, frame)
                for date, state in zip(dates, states)
            ]
        )

    @staticmethod
    def _from_cartesian(
        date: datetime | pd.Timestamp,
        state: np.ndarray,
        angle: AngleType,
        mu: float = Constants.DEFAULT_MU,
        frame: Frame = Constants.DEFAULT_ECI,
    ) -> np.ndarray:
        # Convert date and state to Orekit format
        dat = datetime_to_absolutedate(date)
        pos = Vector3D(*state[0:3].tolist())
        vel = Vector3D(*state[3:6].tolist())

        # Create spacecraft state
        pv = TimeStampedPVCoordinates(dat, pos, vel)

        # Convert to Keplerian state
        keplerian = KeplerianOrbit(pv, frame, mu)

        # Extract Keplerian elements
        # NOTE: angles are wrapped to [0, 2pi)
        a = keplerian.getA()
        e = keplerian.getE()
        i = keplerian.getI()
        raan = keplerian.getRightAscensionOfAscendingNode() % (2.0 * np.pi)
        aop = keplerian.getPerigeeArgument() % (2.0 * np.pi)
        ta = keplerian.getTrueAnomaly() % (2.0 * np.pi)
        ma = keplerian.getMeanAnomaly() % (2.0 * np.pi)
        ea = keplerian.getEccentricAnomaly() % (2.0 * np.pi)

        # Return extracted Keplerian elements
        if angle == AngleType.TRUE:
            return np.array([a, e, i, raan, aop, ta])
        elif angle == AngleType.MEAN:
            return np.array([a, e, i, raan, aop, ma])
        elif angle == AngleType.ECCENTRIC:
            return np.array([a, e, i, raan, aop, ea])
        else:
            raise RuntimeError("Unknown angle type")

    @staticmethod
    def to_cartesian(
        dates: List[datetime] | pd.DatetimeIndex,
        states: np.ndarray,
        angle: AngleType = AngleType.MEAN,
        mu: float = Constants.DEFAULT_MU,
        frame: Frame = Constants.DEFAULT_ECI,
    ) -> np.ndarray:
        # Return Cartesian states
        return np.array(
            [
                Keplerian._to_cartesian(date, state, angle, mu, frame)
                for date, state in zip(dates, states)
            ]
        )

    @staticmethod
    def _to_cartesian(
        date: datetime | pd.Timestamp,
        state: np.ndarray,
        angle: AngleType,
        mu: float = Constants.DEFAULT_MU,
        frame: Frame = Constants.DEFAULT_ECI,
    ) -> np.ndarray:
        # Convert date to Orekit format
        dat = datetime_to_absolutedate(date)

        # Extract Keplerian elements
        a, e, i, raan, aop, an = state

        # Ensure that the variables are floats
        a = float(a)
        e = float(e)
        i = float(i)
        raan = float(raan)
        aop = float(aop)
        an = float(an)

        # Create Keplerian representation
        keplerian = KeplerianOrbit(
            a,
            e,
            i,
            aop,
            raan,
            an,
            angle.value,
            frame,
            dat,
            mu,
        )

        # Extract position and velocity
        pv = keplerian.getPVCoordinates()
        pos = pv.getPosition().toArray()
        vel = pv.getVelocity().toArray()

        # Return extracted Cartesian state
        return np.array([*pos, *vel])
