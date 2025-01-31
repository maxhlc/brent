# Standard imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
from skyfield.api import Loader, utc
from skyfield.elementslib import osculating_elements_of

# Internal imports
from .bias import Bias
from .factory import BiasFactory
from brent.frames import RTN
from brent.paths import SKYFIELD_DIR


@BiasFactory.register("moonanomaly")
@dataclass
class MoonAnomalyBias(Bias):
    # Model parameters
    amplitude: float
    phase: float
    offset: float

    def _model(self, ma: np.ndarray) -> np.ndarray:
        # Return along-track bias
        return self.amplitude * np.sin(ma + self.phase) + self.offset

    @classmethod
    def _mean_anomaly(cls, dates) -> np.ndarray:
        # Generate dates
        # TODO: enforce UTC at a project level?
        dates_ = [date.replace(tzinfo=utc) for date in dates]
        ts = cls.LOADER.timescale()
        t = ts.from_datetimes(dates_)

        # Extract Earth and Moon from loaded ephemerides
        earth = cls.OBJECTS["earth"]
        moon = cls.OBJECTS["moon"]

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

    # Load ephemerides
    LOADER = Loader(SKYFIELD_DIR)
    OBJECTS = LOADER("de421.bsp")
