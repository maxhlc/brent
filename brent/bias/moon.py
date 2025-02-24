# Standard imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
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
