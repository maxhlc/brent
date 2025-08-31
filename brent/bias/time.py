# Future imports
from __future__ import annotations

# Standard imports
from typing import overload
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import numpy as np
import scipy.fft
import scipy.optimize

# Internal imports
from .factory import BiasFactory
from .bias import Bias
from brent.frames import RTN, Keplerian
from brent.util import Wrap

# Reference epoch when converting dates to days
# NOTE: approximately J2000 (small difference between UTC and TT)
EPOCH_J2000 = datetime(2000, 1, 1, 12, 0, 0, 0)


@BiasFactory.register("time_position")
class TimePositionBias(Bias):

    @overload
    def __init__(self, a: float, b: float, c: float, d: float): ...

    @overload
    def __init__(self, a: list[float], b: list[float], c: list[float], d: float): ...

    @overload
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: float): ...

    def __init__(
        self,
        a: float | list[float] | np.ndarray,
        b: float | list[float] | np.ndarray,
        c: float | list[float] | np.ndarray,
        d: float,
    ):
        # Create parameter arrays
        a_ = np.array(a).ravel()
        b_ = np.array(b).ravel()
        c_ = np.array(c).ravel()

        # Ensure all parameter arrays are the same size
        assert a_.size == b_.size == c_.size

        # Store parameters
        self.a = a_
        self.b = b_
        self.c = c_
        self.d = d

    def _model(self, t: np.ndarray) -> np.ndarray:
        # Extract model parameters
        a = self.a.reshape((-1, 1))
        b = self.b.reshape((-1, 1))
        c = self.c.reshape((-1, 1))
        d = self.d

        # Calculate bias frequencies
        frequency = 2.0 * np.pi / b

        # Calculate bias components
        periodics = a * np.cos(frequency * (t + c))

        # Calculate total bias
        bias = np.sum(periodics, axis=0) + d

        # Return along-track bias
        return bias

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - EPOCH_J2000)

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self._model(t) * rmag

        # Rotate to inertial frame
        bias = RTN.transform(rtn, bias_RTN, reverse=True)

        # Return biases
        return bias

    @classmethod
    def _wrap(
        cls,
        a: float,
        b: float,
        c: float,
        d: float,
    ) -> tuple[float, float, float, float]:
        # Check for negative amplitude
        if a < 0.0:
            # Flip amplitude sign
            a *= -1.0

            # Update phase by half period
            c += 0.5 * b

        # Wrap phase by period
        c = Wrap.half(c, b)

        # Return wrapped parameters
        return a, b, c, d

    @classmethod
    def fit(cls, dates, states, reference, p0, p_scale) -> TimePositionBias:
        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - EPOCH_J2000)

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

    @classmethod
    def fit_fft(
        cls,
        dates: np.ndarray,
        states: np.ndarray,
        reference: np.ndarray,
        minperiod: float = 1.0,
        n: int = 10,
    ) -> TimePositionBias:
        # Sort dates and states
        # TODO: make copies?
        idx_sort = np.argsort(dates)
        dates = dates[idx_sort]
        states = states[idx_sort, :]
        reference = reference[idx_sort, :]

        # Ensure dates are evenly-spaced
        timedeltas = np.diff(dates)
        if len(np.unique(timedeltas)) != 1:
            raise RuntimeError("Variable spacing in dates")

        # Calculate along-track error
        rmag = np.linalg.norm(states[:, 0:3], axis=1)
        rtn = RTN.getTransform(states)
        delta = -(reference - states)
        along_track = RTN.transform(rtn, delta)[:, 1] / rmag

        # Extract along-track error
        t = along_track

        # Remove mean offset
        offset = np.mean(along_track)
        y = t - offset

        # Extract number of samples and timestep
        N = len(y)
        T = timedeltas[0] / np.timedelta64(1, "D")

        # Calculate FFT
        yf = scipy.fft.fft(y)

        # Calculate amplitudes, frequencies, and phases
        amplitudes = 2.0 / N * np.abs(yf)
        frequencies = scipy.fft.fftfreq(N, T)
        phases_ = np.angle(yf)

        # Calculate periods (with NaN for constant term)
        periods = np.concat((np.array([np.nan]), 1 / frequencies[1:]))

        # Calculate phases at reference epoch (J2000)
        j2000_offset = (dates[0] - np.datetime64(EPOCH_J2000)) / np.timedelta64(1, "D")
        phases = (phases_ / (2 * np.pi) * periods) - j2000_offset

        # Wrap phases
        # TODO: make nicer version of wrap for multiple periods
        phases = np.array([Wrap.half(iph, ipe) for iph, ipe in zip(phases, periods)])

        # Filters
        # NOTE: more filters could be added here
        # TODO: make minperiod a timedelta?
        idx_period = periods[0 : len(amplitudes) // 2] >= minperiod

        # Combine filters
        (idx,) = np.where(idx_period)

        # Find largest components
        idx_largest = idx[np.argsort(amplitudes[idx])][-1 : -1 - n : -1]

        # Sort components by period
        idx_largest = idx_largest[np.argsort(periods[idx_largest])]

        # Return fitted bias model
        return TimePositionBias(
            amplitudes[idx_largest],
            periods[idx_largest],
            phases[idx_largest],
            offset,
        )


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

    # TODO: update to accept multiple time-dependent components
    # TODO: could make the various components callables?

    def _model(self, t: np.ndarray, raan: np.ndarray) -> np.ndarray:
        # Calculate model period
        frequency = 2.0 * np.pi / self.b

        # Calculate amplitudes
        amplitude = self.e * np.cos(raan + self.f) + self.g

        # Return along-track bias
        return amplitude * np.cos(frequency * (t + self.c)) + self.d

    def biases(self, dates, states) -> np.ndarray:
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        rtn = RTN.getTransform(states)

        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - EPOCH_J2000)

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
    def _wrap(
        cls,
        b: float,
        c: float,
        d: float,
        e: float,
        f: float,
        g: float,
    ) -> tuple[float, float, float, float, float, float]:
        # Wrap phase by period
        c = Wrap.half(c, b)

        # Check for negative amplitude
        if e < 0.0:
            # Flip amplitude sign
            e *= -1.0

            # Update phase by half period
            f += np.pi

        # Wrap phase by period
        f = Wrap.half(f)

        # Return wrapped parameters
        return b, c, d, e, f, g

    @classmethod
    def fit(cls, dates, states, reference, p0, p_scale) -> TimePositionCombinedBias:
        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - EPOCH_J2000)

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
