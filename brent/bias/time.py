# Standard imports
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import numpy as np

# Internal imports
from .factory import BiasFactory
from .bias import Bias
from brent.frames import RTN, Keplerian


@BiasFactory.register("time")
@dataclass
class TimeBias(Bias):
    # Model parameters
    amplitude: float
    frequency: float
    phase: float
    offset: float

    def _model(self, t: np.ndarray) -> np.ndarray:
        # Calculate model period
        period = 2.0 * np.pi / self.frequency

        # Return along-track bias
        return self.amplitude * np.sin(period * (t + self.phase)) + self.offset

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - self.REFERENCE_EPOCH)

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self._model(t) * rmag

        # Rotate to inertial frame
        bias = RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias

    # Reference epoch when converting dates to days
    REFERENCE_EPOCH = datetime(2000, 1, 1, 0, 0, 0, 0)


@BiasFactory.register("timecombined")
@dataclass
class TimeCombinedBias(Bias):
    # Model parameters
    frequency: float
    phase: float
    offset: float
    d: float
    e: float
    f: float

    def _model(self, t: np.ndarray, raan: np.ndarray) -> np.ndarray:
        # Calculate model period
        period = 2.0 * np.pi / self.frequency

        # Return along-track bias
        return (self.d * np.sin(raan + self.e) + self.f) * np.sin(period * (t + self.phase)) + self.offset

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - self.REFERENCE_EPOCH)

        # Calculate RAAN of object
        keplerian = Keplerian.from_cartesian(dates, states)
        raan = keplerian[:, 3]

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self._model(t, raan) * rmag

        # Rotate to inertial frame
        bias = RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias

    # Reference epoch when converting dates to days
    REFERENCE_EPOCH = datetime(2000, 1, 1, 0, 0, 0, 0)
