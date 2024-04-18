# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate

# Internal imports
from brent import Constants


class Propagator:
    def __init__(self, propagator):
        # Store propagator
        self.propagator = propagator

    def propagate(self, dates, frame=Constants.DEFAULT_ECI):
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
