# Standard imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
import scipy.optimize
from skyfield.api import utc
from skyfield.elementslib import osculating_elements_of

# Internal imports
from .bias import Bias
from .factory import BiasFactory
from brent.frames import RTN, Keplerian
from brent.skyfield import Skyfield


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

    @classmethod
    def _mean_anomaly(cls, dates) -> np.ndarray:
        # Generate dates
        # TODO: enforce UTC at a project level?
        dates_ = [date.replace(tzinfo=utc) for date in dates]
        ts = Skyfield.LOADER.timescale()
        t = ts.from_datetimes(dates_)

        # Extract Earth and Moon from loaded ephemerides
        earth = Skyfield.OBJECTS["earth"]
        moon = Skyfield.OBJECTS["moon"]

        # Calculate Moon state and orbital elements
        moon_state = (moon - earth).at(t)
        moon_elements = osculating_elements_of(moon_state)

        # Extract mean anomaly
        ma = moon_elements.mean_anomaly.radians

        # Return mean anomaly
        return ma

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate mean anomaly of the Moon
        ma = self._mean_anomaly(dates)

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self._model(ma) * rmag

        # Rotate to inertial frame
        bias = RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias

    @classmethod
    def fit(cls, dates, states, reference, p0, p_scale) -> Bias:
        # Calculate Moon's mean anomaly
        ma = cls._mean_anomaly(dates)

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
        # Return along-track bias
        return (self.e * np.sin(raan + self.f) + self.g) * np.sin(ma + self.c) + self.d

    @classmethod
    def _mean_anomaly(cls, dates) -> np.ndarray:
        # Generate dates
        # TODO: enforce UTC at a project level?
        dates_ = [date.replace(tzinfo=utc) for date in dates]
        ts = Skyfield.LOADER.timescale()
        t = ts.from_datetimes(dates_)

        # Extract Earth and Moon from loaded ephemerides
        earth = Skyfield.OBJECTS["earth"]
        moon = Skyfield.OBJECTS["moon"]

        # Calculate Moon state and orbital elements
        moon_state = (moon - earth).at(t)
        moon_elements = osculating_elements_of(moon_state)

        # Extract mean anomaly
        ma = moon_elements.mean_anomaly.radians

        # Return mean anomaly
        return ma

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate mean anomaly of the Moon
        ma = self._mean_anomaly(dates)

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
    def fit(cls, dates, states, reference, p0, p_scale) -> Bias:
        # Calculate Moon's mean anomaly
        ma = cls._mean_anomaly(dates)

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
