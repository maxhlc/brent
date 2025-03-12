# Standard imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.propagation import Propagator as OrekitPropagator

# Internal imports
from brent import Constants


class Propagator(ABC):

    @abstractmethod
    def _propagate(self, date, frame=Constants.DEFAULT_ECI) -> np.ndarray: ...

    def _wrapped_propagate(self, date, frame=Constants.DEFAULT_ECI) -> np.ndarray:
        try:
            # Return propagated state
            return self._propagate(date, frame)
        except:
            # Return NaN state
            return np.array(6 * [np.nan])

    def propagate(self, dates, frame=Constants.DEFAULT_ECI) -> np.ndarray:
        # Return states array
        return np.array([self._wrapped_propagate(date, frame) for date in dates])


class WrappedPropagator(Propagator):

    def __init__(self, propagator: OrekitPropagator) -> None:
        # Store propagator
        self.propagator = propagator

    def _propagate(self, date, frame=Constants.DEFAULT_ECI):
        # Convert date to Orekit format
        date_ = datetime_to_absolutedate(date)

        # Propagate state
        state_ = self.propagator.getPVCoordinates(date_, frame)

        # Extract position and velocity
        pos = state_.getPosition().toArray()
        vel = state_.getVelocity().toArray()

        # Concatenate position and velocity
        state = np.array(pos + vel)

        # Return state vector
        return state
