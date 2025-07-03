# Future imports
from __future__ import annotations

# Standard imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
import scipy.optimize

# Internal imports
from .bias import Bias
from .factory import BiasFactory
from brent.frames import RTN, Keplerian
from brent.skyfield import Earth, Moon
from brent.util import Wrap


@BiasFactory.register("moonanomaly_position")
@dataclass
class MoonAnomalyPositionBias(Bias):
    # Model parameters
    a: float
    c: float
    d: float

    def _model(self, ma: np.ndarray) -> np.ndarray:
        # Return along-track bias
        return self.a * np.sin(ma + self.c) + self.d

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate mean anomaly of the Moon
        ma = Moon.keplerian(dates, Earth)[:, 5]

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self._model(ma) * rmag

        # Rotate to inertial frame
        bias = RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias

    @classmethod
    def _wrap(cls, a: float, c: float, d: float) -> tuple[float, float, float]:
        # Check for negative amplitude
        if a < 0.0:
            # Flip amplitude sign
            a *= -1.0

            # Update phase by half period
            c += np.pi

        # Wrap phase by period
        c = Wrap.half(c)

        # Return wrapped parameters
        return a, c, d

    @classmethod
    def fit(cls, dates, states, reference, p0, p_scale) -> MoonAnomalyPositionBias:
        # Calculate Moon's mean anomaly
        ma = Moon.keplerian(dates, Earth)[:, 5]

        # Fit wrapper function
        def func(_, *p):
            # Scale parameters
            params = np.array(p) * p_scale

            # Create bias model
            model = cls(*params)

            # Estimate along-track bias
            along_track_ = model._model(ma)

            # Return along-track bias
            return along_track_.ravel()

        # Calculate along-track error
        rmag = np.linalg.norm(states[:, 0:3], axis=1)
        rtn = RTN.getTransform(states)
        delta = -(reference - states)
        along_track = RTN.transform(rtn, delta)[:, 1] / rmag

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

        # Wrap parameters
        params = cls._wrap(*params)

        # Returned fitted bias model
        return cls(*params)


@BiasFactory.register("moonanomaly_position_combined")
@dataclass
class MoonAnomalyPositionCombinedBias(Bias):
    # Model parameters
    c: float
    d: float
    e: float
    f: float
    g: float

    def _model(self, ma: np.ndarray, raan: np.ndarray) -> np.ndarray:
        # Calculate amplitude
        amplitude = self.e * np.sin(raan + self.f) + self.g

        # Return along-track bias
        return amplitude * np.sin(ma + self.c) + self.d

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate mean anomaly of the Moon
        ma = Moon.keplerian(dates, Earth)[:, 5]

        # Calculate RAAN of object
        keplerian = Keplerian.from_cartesian(dates, states)
        raan = keplerian[:, 3]

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self._model(ma, raan) * rmag

        # Rotate to inertial frame
        bias = RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias

    @classmethod
    def _wrap(
        cls,
        c: float,
        d: float,
        e: float,
        f: float,
        g: float,
    ) -> tuple[float, float, float, float, float]:
        # Wrap phase by period
        c = Wrap.half(c)

        # Check for negative amplitude
        if e < 0.0:
            # Flip amplitude sign
            e *= -1.0

            # Update phase by half period
            f += np.pi

        # Wrap phase by period
        f = Wrap.half(f)

        # Return wrapped parameters
        return c, d, e, f, g

    @classmethod
    def fit(
        cls,
        dates,
        states,
        reference,
        p0,
        p_scale,
    ) -> MoonAnomalyPositionCombinedBias:
        # Calculate Moon's mean anomaly
        ma = Moon.keplerian(dates, Earth)[:, 5]

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
            along_track_ = model._model(ma, raan)

            # Return along-track bias
            return along_track_.ravel()

        # Calculate along-track error
        rmag = np.linalg.norm(states[:, 0:3], axis=1)
        rtn = RTN.getTransform(states)
        delta = -(reference - states)
        along_track = RTN.transform(rtn, delta)[:, 1] / rmag

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

        # Wrap parameters
        params = cls._wrap(*params)

        # Returned fitted bias model
        return cls(*params)
