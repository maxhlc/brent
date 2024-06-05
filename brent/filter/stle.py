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

    # Decision vector scaling
    XSCALE = np.array(
        [
            2.0,  # n
            0.001,  # e
            np.pi,  # i
            2 * np.pi,  # raan
            2 * np.pi,  # aop
            2 * np.pi,  # ma
            0.0001,  # bstar
        ]
    )

    # Decision vector bounds
    XBOUNDS = [
        [0.0, np.inf],  # n
        [0.0, 1.0],  # e
        [-np.inf, np.inf],  # i
        [-np.inf, np.inf],  # raan
        [-np.inf, np.inf],  # aop
        [-np.inf, np.inf],  # ma
        [-1e-3, 1e-3],  # bstar
    ]

    def __init__(
        self,
        dates,
        states: np.ndarray,
        reference_index: int = -1,
        estimate_bstar: bool = True,
    ):
        # Store copies of the dates and states
        self.dates = deepcopy(dates)
        self.states = deepcopy(states)
        self.reference_index = reference_index

        # Store BSTAR flag
        self.estimate_bstar = estimate_bstar

        # Extract reference date and state
        self.reference_date = dates[reference_index]
        self.reference_state = states[reference_index, :]

        # Convert final date and state to Orekit format
        # TODO: repeated across codebase, move to separate function
        date_ = datetime_to_absolutedate(self.reference_date)
        pos_ = Vector3D(*self.reference_state[0:3].tolist())
        vel_ = Vector3D(*self.reference_state[3:6].tolist())
        state_ = TimeStampedPVCoordinates(date_, pos_, vel_)
        orbit = CartesianOrbit(state_, Constants.DEFAULT_ECI, Constants.DEFAULT_MU)

        # Create template TLE
        templateTLE = SyntheticTLEGenerator.__generate_tle(date=self.reference_date)

        # Generate initial TLE
        initialTLE = TLE.stateToTLE(
            SpacecraftState(orbit),
            templateTLE,
            FixedPointTleGenerationAlgorithm(),
        )

        # Generate initial guess
        self.x0 = SyntheticTLEGenerator._tle_to_vector(initialTLE, estimate_bstar)

    def estimate(self) -> TLEPropagator:
        # Load dates and states
        dates = self.dates
        reference_date = self.reference_date
        reference_index = self.reference_index
        states = self.states

        # Load initial guess
        x0 = self.x0

        # Calculate position and velocity magnitudes
        r0 = np.linalg.norm(states[reference_index, 0:3])
        v0 = np.linalg.norm(states[reference_index, 3:6])

        # Calculate scaling units
        lu = 1.0 / (2.0 / r0 - v0**2 / Constants.DEFAULT_MU)
        vu = np.sqrt(Constants.DEFAULT_MU / lu)
        fscale = np.array([[lu, lu, lu, vu, vu, vu]])

        # Define residual function
        def fun(x):
            # Generate TLE
            tle = SyntheticTLEGenerator._vector_to_tle(reference_date, x)

            # Propagate TLE
            states_ = TLEPropagator([tle]).propagate(dates)

            # Calculate state error
            delta = states_ - states

            # Scale state error
            delta /= fscale

            # Return vector of errors
            return delta.ravel()

        # Set decision vector scaling and optimisation bounds
        if self.estimate_bstar:
            xscale = SyntheticTLEGenerator.XSCALE
            bounds = SyntheticTLEGenerator.XBOUNDS
        else:
            xscale = SyntheticTLEGenerator.XSCALE[:6]
            bounds = SyntheticTLEGenerator.XBOUNDS[:6]

        # Reorganise bounds to expected format
        bounds = list(zip(*bounds))

        # Execute optimiser
        sol = scipy.optimize.least_squares(
            fun,
            x0,
            x_scale=xscale,
            bounds=bounds,
        )

        # Create TLE
        tle = SyntheticTLEGenerator._vector_to_tle(reference_date, sol.x)

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
        # Return TLE
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
    def _tle_to_vector(tle: TLE, bstar: bool) -> np.ndarray:
        # Extract elements
        elements = [
            tle.getMeanMotion(),
            tle.getE(),
            tle.getI(),
            tle.getRaan(),
            tle.getPerigeeArgument(),
            tle.getMeanAnomaly(),
        ]

        # Add BSTAR (if estimated)
        if bstar:
            elements.append(tle.getBStar())

        # Return variables
        return np.array(elements)

    @staticmethod
    def _vector_to_tle(date: datetime, elements: np.ndarray) -> TLE:
        # Extract variables
        if len(elements) == 7:
            n, e, i, raan, aop, ma, bstar = elements
        elif len(elements) == 6:
            n, e, i, raan, aop, ma = elements
            bstar = 0.0
        else:
            raise ValueError("Incompatible elements vector length")

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
