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
from org.orekit.orbits import EquinoctialOrbit
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from .angle import AngleType
from brent import Constants


class Equinoctial:

    @staticmethod
    def from_cartesian(
        dates: List[datetime] | pd.DatetimeIndex,
        states: np.ndarray,
        angle: AngleType = AngleType.MEAN,
        mu: float = Constants.DEFAULT_MU,
        frame: Frame = Constants.DEFAULT_ECI,
    ) -> np.ndarray:
        # Return Equinoctial elementss
        return np.array(
            [
                Equinoctial._from_cartesian(date, state, angle, mu, frame)
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

        # Convert to Equinoctial state
        equinoctial = EquinoctialOrbit(pv, frame, mu)

        # Extract Equinoctial elements
        # NOTE: angles are wrapped to [0, 2pi)
        a = equinoctial.getA()
        ex = equinoctial.getEquinoctialEx()
        ey = equinoctial.getEquinoctialEy()
        hx = equinoctial.getHx()
        hy = equinoctial.getHy()
        lt = equinoctial.getLv() % (2.0 * np.pi)  # True
        lm = equinoctial.getLM() % (2.0 * np.pi)  # Mean
        le = equinoctial.getLE() % (2.0 * np.pi)  # Eccentric

        # Return extracted Equinoctial elements
        if angle == AngleType.TRUE:
            return np.array([a, ex, ey, hx, hy, lt])
        elif angle == AngleType.MEAN:
            return np.array([a, ex, ey, hx, hy, lm])
        elif angle == AngleType.ECCENTRIC:
            return np.array([a, ex, ey, hx, hy, le])
        else:
            raise RuntimeError("Unknown angle type")

    @staticmethod
    def to_cartesian(
        dates: List[datetime] | pd.DatetimeIndex,
        states: np.ndarray,
        angle: AngleType = AngleType.MEAN,
        mu: float = Constants.DEFAULT_MU,
        frame: Frame = Constants.DEFAULT_ECI,
    ):
        # Return Cartesian states
        return np.array(
            [
                Equinoctial._to_cartesian(date, state, angle, mu, frame)
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
    ):
        # Convert date to Orekit format
        dat = datetime_to_absolutedate(date)

        # Extract Equinoctial elements
        a, ex, ey, hx, hy, lon = state

        # Ensure that the variables are floats
        a = float(a)
        ex = float(ex)
        ey = float(ey)
        hx = float(hx)
        hy = float(hy)
        lon = float(lon)

        # Create Equinoctial representation
        equinoctial = EquinoctialOrbit(
            a,
            ex,
            ey,
            hx,
            hy,
            lon,
            angle.value,
            frame,
            dat,
            mu,
        )

        # Extract position and velocity
        pv = equinoctial.getPVCoordinates()
        pos = pv.getPosition().toArray()
        vel = pv.getVelocity().toArray()

        # Return extracted Cartesian state
        return np.array([*pos, *vel])
