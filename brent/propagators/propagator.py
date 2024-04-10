# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.orbits import CartesianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from .constants import DEFAULT_ECI, DEFAULT_MU


class Propagator:
    def __init__(self, propagator):
        # Store propagator
        self.propagator = propagator

    def setInitialState(self, date, state, frame=DEFAULT_ECI, mu=DEFAULT_MU):
        # Convert initial date and state to Orekit format
        date_ = datetime_to_absolutedate(date)
        pos = Vector3D(*state[0:3].tolist())
        vel = Vector3D(*state[3:6].tolist())

        # Generate new spacecraft state
        state = TimeStampedPVCoordinates(date_, pos, vel)
        orbit = CartesianOrbit(state, frame, mu)
        spacecraftState = SpacecraftState(orbit)

        # Update propagator initial state
        self.propagator.setInitialState(spacecraftState)

    def propagate(self, dates, frame=DEFAULT_ECI):
        # Declare states array
        states = np.empty((len(dates), 6))
        states.fill(np.nan)

        # Iterate through dates
        for idx, date in enumerate(dates):
            # Convert date to Orekit format
            date_ = datetime_to_absolutedate(date)

            # Propagate state
            state = self.propagator.getPVCoordinates(date_, frame)

            # Extract position and velocity
            pos = state.getPosition().toArray()
            vel = state.getVelocity().toArray()

            # Update state matrix
            states[idx, 0:3] = pos
            states[idx, 3:6] = vel

        # Return states array
        return states
