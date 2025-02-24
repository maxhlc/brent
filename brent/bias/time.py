# Standard imports
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import numpy as np
import scipy.optimize

# Internal imports
from .factory import BiasFactory
from .bias import Bias
from brent.frames import RTN, Keplerian


@BiasFactory.register("time_position")
@dataclass
class TimePositionBias(Bias):
    # Model parameters
    a: float
    b: float
    c: float
    d: float

    def _model(self, t: np.ndarray) -> np.ndarray:
        # Calculate model period
        frequency = 2.0 * np.pi / self.b

        # Return along-track bias
        return self.a * np.sin(frequency * (t + self.c)) + self.d

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

    @classmethod
    def fit(cls, dates, states, reference, p0, p_scale) -> Bias:
        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - cls.REFERENCE_EPOCH)

        # Fit wrapper function
        def func(_, *p):
            # Scale parameters
            params = np.array(p) * p_scale

            # Create bias model
            model = cls(*params)

            # Estimate along-track bias
            along_track_ = model._model(t)

            # Return along-track bias
            return along_track_.ravel()

        # Calculate along-track error
        rmag = np.linalg.norm(states[:, 0:3], axis=1)
        rtn = RTN.getTransform(states)
        along_track = RTN.transform(rtn, states - reference)[:, 1] / rmag

        # Extract fit data
        y = along_track.ravel()
        x = np.zeros(y.shape)

        # Fit model
        popt, _ = scipy.optimize.curve_fit(
            f=func,
            xdata=x,
            ydata=y,
            p0=p0 / p_scale,
        )

        # Scale parameters
        params = popt * p_scale

        # Returned fitted bias model
        return cls(*params)

    # Reference epoch when converting dates to days
    REFERENCE_EPOCH = datetime(2000, 1, 1, 0, 0, 0, 0)


@BiasFactory.register("time_position_combined")
@dataclass
class TimePositionCombinedBias(Bias):
    # Model parameters
    b: float
    c: float
    d: float
    e: float
    f: float
    g: float

    def _model(self, t: np.ndarray, raan: np.ndarray) -> np.ndarray:
        # Calculate model period
        frequency = 2.0 * np.pi / self.b

        # Return along-track bias
        return (self.e * np.sin(raan + self.f) + self.g) * np.sin(frequency * (t + self.c)) + self.d

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

    @classmethod
    def fit(cls, dates, states, reference, p0, p_scale) -> Bias:
        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - cls.REFERENCE_EPOCH)

        # Calculate RAAN of object
        keplerian = Keplerian.from_cartesian(dates, states)
        raan = keplerian[:, 3]

        # Fit wrapper function
        def func(_, *p):
            # Scale parameters
            params = np.array(p) * p_scale

            # Create bias model
            model = cls(*params)

            # Estimate along-track bias
            along_track_ = model._model(t, raan)

            # Return along-track bias
            return along_track_.ravel()

        # Calculate along-track error
        rmag = np.linalg.norm(states[:, 0:3], axis=1)
        rtn = RTN.getTransform(states)
        along_track = RTN.transform(rtn, states - reference)[:, 1] / rmag

        # Extract fit data
        y = along_track.ravel()
        x = np.zeros(y.shape)

        # Fit model
        popt, _ = scipy.optimize.curve_fit(
            f=func,
            xdata=x,
            ydata=y,
            p0=p0 / p_scale,
        )

        # Scale parameters
        params = popt * p_scale

        # Returned fitted bias model
        return cls(*params)

    # Reference epoch when converting dates to days
    REFERENCE_EPOCH = datetime(2000, 1, 1, 0, 0, 0, 0)
