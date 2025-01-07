# Standard imports
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import numpy as np

# Internal imports
from .factory import BiasFactory
from .bias import Bias
import brent.frames


@BiasFactory.register("simplifiedalongtracksinusoidal")
@dataclass
class SimplifiedAlongtrackSinusoidalBias(Bias):
    # Model parameters
    amplitude: float
    frequency: float
    phase: float
    offset: float

    def __model(self, t: np.ndarray) -> np.ndarray:
        # Calculate model period
        period = 2.0 * np.pi / self.frequency

        # Return along-track bias
        return self.amplitude * np.sin(period * (t + self.phase)) + self.offset

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = brent.frames.RTN.getTransform(states)

        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - datetime(2022, 1, 1))

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self.__model(t) * rmag

        # Rotate to inertial frame
        bias = brent.frames.RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias
