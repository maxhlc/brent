# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.orbits import EquinoctialOrbit, PositionAngleType
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from brent import Constants


def cartesian_to_equinoctial(
    dates,
    states,
    mu=Constants.DEFAULT_MU,
    frame=Constants.DEFAULT_ECI,
):
    def _cartesian_to_equinoctial(date, state):
        if isinstance(date, np.datetime64):
            date = pd.Timestamp(date)

        # Convert date and state to Orekit format
        dat = datetime_to_absolutedate(date)
        pos = Vector3D(*state[0:3].tolist())
        vel = Vector3D(*state[3:6].tolist())

        # Create spacecraft state
        pv = TimeStampedPVCoordinates(dat, pos, vel)

        # Convert to Equinoctial state
        equinoctial = EquinoctialOrbit(pv, frame, mu)

        # Extract Equinoctial elements
        # TODO: angle type as input
        a = equinoctial.getA()
        ex = equinoctial.getEquinoctialEx()
        ey = equinoctial.getEquinoctialEy()
        hx = equinoctial.getHx()
        hy = equinoctial.getHy()
        lm = equinoctial.getLM()

        # Return extracted Equinoctial elements
        return [a, ex, ey, hx, hy, lm]

    # Return Equinoctial elements
    return np.array(
        [_cartesian_to_equinoctial(date, state) for date, state in zip(dates, states)]
    )


def equinoctial_to_cartesian(
    dates,
    states,
    mu=Constants.DEFAULT_MU,
    frame=Constants.DEFAULT_ECI,
):
    def _equinoctial_to_cartesian(date, state):
        if isinstance(date, np.datetime64):
            date = pd.Timestamp(date)

        # Convert date to Orekit format
        dat = datetime_to_absolutedate(date)

        # Extract Equinoctial elements
        a, ex, ey, hx, hy, lm = state

        # Ensure that the variables are floats
        a = float(a)
        ex = float(ex)
        ey = float(ey)
        hx = float(hx)
        hy = float(hy)
        lm = float(lm)

        # Create Equinoctial representation
        # TODO: angle type as input
        equinoctial = EquinoctialOrbit(
            a,
            ex,
            ey,
            hx,
            hy,
            lm,
            PositionAngleType.MEAN,
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

    # Return Cartesian states
    return np.array(
        [_equinoctial_to_cartesian(date, state) for date, state in zip(dates, states)]
    )
