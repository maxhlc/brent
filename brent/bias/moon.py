# Third-party imports
import numpy as np
from skyfield.api import Loader, utc
from skyfield.elementslib import osculating_elements_of

# Internal imports
from .bias import Bias
from .factory import BiasFactory
from brent.frames import RTN, Keplerian
from brent.paths import SKYFIELD_DIR


def poly(x, *args):
    return np.sum(np.array([arg * x**ii for ii, arg in enumerate(args)]), axis=0)


@BiasFactory.register("moonanomaly")
class MoonAnomalyBias(Bias):

    @classmethod
    def _model(cls, ma: np.ndarray, amplitude, phase, offset) -> np.ndarray:
        # Return along-track bias
        return amplitude * np.sin(ma + phase) + offset

    @classmethod
    def _mean_anomaly(cls, dates) -> np.ndarray:
        # Generate dates
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

        # Calculate object RAAN
        # TODO: specify gravitational parameter and frame
        kep = Keplerian.from_cartesian(dates, states)
        raan = kep[:, 3]

        # Calculate model parameters
        amplitude = poly(raan, *self.POLY_COEFF)
        phase = 0.0
        offset = 0.0

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self._model(ma, amplitude, phase, offset) * rmag

        # Rotate to inertial frame
        bias = RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias

    # Load ephemerides
    LOADER = Loader(SKYFIELD_DIR)
    OBJECTS = LOADER("de421.bsp")

    # Amplitude-RAAN polynomial coefficients
    POLY_COEFF = np.array([9.61421421e-05, -3.11943960e-05, 4.75645018e-06])
