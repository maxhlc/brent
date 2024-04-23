# Standard imports
from copy import deepcopy
from datetime import datetime

# Third-party imports
import numpy as np
import scipy.optimize

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical.tle import TLE
from org.orekit.propagation.analytical.tle.generation import (
    FixedPointTleGenerationAlgorithm,
)
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from brent import Constants
from brent.propagators import TLEPropagator


class SyntheticTLEGenerator:

    def __init__(self, dates, states):
        # Store copies of the dates and states
        self.dates = deepcopy(dates)
        self.states = deepcopy(states)

        # Convert final date and state to Orekit format
        # TODO: repeated across codebase, move to separate function
        date_ = datetime_to_absolutedate(dates[-1])
        pos_ = Vector3D(*states[-1, 0:3].tolist())
        vel_ = Vector3D(*states[-1, 3:6].tolist())
        state_ = TimeStampedPVCoordinates(date_, pos_, vel_)
        orbit = CartesianOrbit(state_, Constants.DEFAULT_ECI, Constants.DEFAULT_MU)

        # Create template TLE
        templateTLE = SyntheticTLEGenerator.__generate_tle(date=dates[-1])

        # Generate initial TLE
        initialTLE = TLE.stateToTLE(
            SpacecraftState(orbit),
            templateTLE,
            FixedPointTleGenerationAlgorithm(),
        )

        # Generate initial guess
        self.x0 = SyntheticTLEGenerator._tle_to_vector(initialTLE)

    def estimate(self) -> TLEPropagator:
        # Load dates and states
        dates = self.dates
        date = dates[-1]
        states = self.states

        # Load initial guess
        x0 = self.x0

        # Calculate position and velocity magnitudes
        r0 = np.linalg.norm(states[-1, 0:3])
        v0 = np.linalg.norm(states[-1, 3:6])

        # Calculate scaling units
        lu = 1.0 / (2.0 / r0 - v0**2 / Constants.DEFAULT_MU)
        vu = np.sqrt(Constants.DEFAULT_MU / lu)
        scaling = np.array([[lu, lu, lu, vu, vu, vu]])

        def fun(x):
            # Generate TLE
            tle = SyntheticTLEGenerator._vector_to_tle(date, x)

            # Propagate TLE
            states_ = TLEPropagator([tle]).propagate(dates)

            # Calculate state error
            delta = states_ - states

            # Scale state error
            delta /= scaling

            # Return vector of errors
            return delta.ravel()

        # Execute optimiser
        sol = scipy.optimize.least_squares(fun, x0, method="lm")

        # Create TLE
        tle = SyntheticTLEGenerator._vector_to_tle(date, sol.x)

        # Return propagator
        return TLEPropagator([tle])

    @staticmethod
    def __generate_tle(
        date: datetime,
        satelliteNumber: int = 0,
        classification: str = " ",
        launchYear: int = 0,
        launchNumber: int = 0,
        launchPiece: str = "   ",
        ephemerisType: int = 0,
        elementNumber: int = 0,
        revolutionNumberAtEpoch: int = 0,
        meanMotion: float = 0.0,
        meanMotionFirstDerivative: float = 0.0,
        meanMotionSecondDerivative: float = 0.0,
        e: float = 0.0,
        i: float = 0.0,
        aop: float = 0.0,
        raan: float = 0.0,
        meanAnomaly: float = 0.0,
        bStar: float = 0.0,
    ) -> TLE:
        # Return TLE template
        return TLE(
            satelliteNumber,
            classification,
            launchYear,
            launchNumber,
            launchPiece,
            ephemerisType,
            elementNumber,
            datetime_to_absolutedate(date),
            meanMotion,
            meanMotionFirstDerivative,
            meanMotionSecondDerivative,
            e,
            i,
            aop,
            raan,
            meanAnomaly,
            revolutionNumberAtEpoch,
            bStar,
        )

    @staticmethod
    def _tle_to_vector(tle: TLE) -> np.ndarray:
        # Return variables
        return np.array(
            [
                tle.getMeanMotion(),
                tle.getE(),
                tle.getI(),
                tle.getRaan(),
                tle.getPerigeeArgument(),
                tle.getMeanAnomaly(),
                tle.getBStar(),
            ]
        )

    @staticmethod
    def _vector_to_tle(date: datetime, var: np.ndarray) -> TLE:
        # Extract variables
        n, e, i, raan, aop, ma, bstar = var

        # Cast variables to correct datatype
        n = float(n)
        e = float(e)
        i = float(i)
        raan = float(raan)
        aop = float(aop)
        ma = float(ma)
        bstar = float(bstar)

        # Return TLE
        return SyntheticTLEGenerator.__generate_tle(
            date=date,
            meanMotion=n,
            e=e,
            i=i,
            aop=aop,
            raan=raan,
            meanAnomaly=ma,
            bStar=bstar,
        )
