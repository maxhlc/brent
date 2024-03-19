# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.orbits import KeplerianOrbit, PositionAngle
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from brent.propagators import DEFAULT_ECI, DEFAULT_MU


def rtn(states):
    # Extract reference position and velocity vectors
    rRef = states[:, 0:3]
    vRef = states[:, 3:6]

    # Calculate reference angular momentum vectors
    hRef = np.cross(rRef, vRef)

    # Calculate magnitudes
    rRefMag = np.linalg.norm(rRef, axis=1, keepdims=True)
    hRefMag = np.linalg.norm(hRef, axis=1, keepdims=True)

    # Calculate RTN components
    R = rRef / rRefMag
    N = hRef / hRefMag
    T = np.cross(N, R)

    # Create RTN matrix
    RTN = np.stack((R, T, N), axis=1)

    # Expand matrix for combined position and velocity rotation
    RTN = np.kron(np.eye(2, dtype=int), RTN)

    # Return transformation matrix
    return RTN


def cartesian_to_keplerian(dates, states, mu=DEFAULT_MU, frame=DEFAULT_ECI):
    def _cartesian_to_keplerian(date, state):
        # Convert date and state to Orekit format
        dat = datetime_to_absolutedate(date)
        pos = Vector3D(*state[0:3].tolist())
        vel = Vector3D(*state[3:6].tolist())

        # Create spacecraft state
        pv = TimeStampedPVCoordinates(dat, pos, vel)

        # Convert to Keplerian state
        keplerian = KeplerianOrbit(pv, frame, mu)

        # Extract Keplerian elements
        # TODO: angle type as input
        a = keplerian.getA()
        e = keplerian.getE()
        i = keplerian.getI()
        raan = keplerian.getRightAscensionOfAscendingNode()
        aop = keplerian.getPerigeeArgument()
        ma = keplerian.getMeanAnomaly()

        # Return extracted Keplerian elements
        return [a, e, i, raan, aop, ma]

    # Return Keplerian elements
    return np.array(
        [_cartesian_to_keplerian(date, state) for date, state in zip(dates, states)]
    )


def keplerian_to_cartesian(dates, states, mu=DEFAULT_MU, frame=DEFAULT_ECI):
    def _keplerian_to_cartesian(date, state):
        # Convert date to Orekit format
        dat = datetime_to_absolutedate(date)

        # Extract Keplerian elements
        a, e, i, raan, aop, ma = state

        # Ensure that the variables are floats
        a = float(a)
        e = float(e)
        i = float(i)
        raan = float(raan)
        aop = float(aop)
        ma = float(ma)

        # Create Keplerian representation
        # TODO: angle type as input
        kep = KeplerianOrbit(a, e, i, aop, raan, ma, PositionAngle.MEAN, frame, dat, mu)

        # Extract position and velocity
        pv = kep.getPVCoordinates()
        pos = pv.getPosition().toArray()
        vel = pv.getVelocity().toArray()

        # Return extracted Cartesian state
        return np.array([*pos, *vel])

    # Return Cartesian states
    return np.array(
        [_keplerian_to_cartesian(date, state) for date, state in zip(dates, states)]
    )
