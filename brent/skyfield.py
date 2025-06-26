# Future imports
from __future__ import annotations

# Standard imports
from typing import Type

# Third-party imports
import numpy as np
from skyfield.api import Loader, utc
from skyfield.elementslib import osculating_elements_of
from skyfield.jpllib import SpiceKernel

# Internal imports
from .paths import SKYFIELD_DIR


class Skyfield:
    # Kernel loader
    LOADER = Loader(SKYFIELD_DIR)

    # Kernel objects
    OBJECTS: SpiceKernel = LOADER("de421.bsp")


class SkyfieldObject:
    # Object name
    name: str

    @classmethod
    def get_object(cls):
        # Return object
        return Skyfield.OBJECTS[cls.name]

    @classmethod
    def states(
        cls,
        dates,
        central: SkyfieldObject | Type[SkyfieldObject],
    ):
        # Generate dates
        # TODO: enforce UTC at a project level?
        dates_ = [date.replace(tzinfo=utc) for date in dates]
        ts = Skyfield.LOADER.timescale()
        t = ts.from_datetimes(dates_)

        # Load ephemerides
        obj1 = cls.get_object()
        obj2 = central.get_object()

        # Calculate states
        states = (obj1 - obj2).at(t)

        # Return states
        return states

    @classmethod
    def keplerian(
        cls,
        dates,
        central: SkyfieldObject | Type[SkyfieldObject],
    ) -> np.ndarray:
        # TODO: anomaly options

        # Calculate states
        states = cls.states(dates, central)

        # Calculate osculating Keplerian elements
        kep = osculating_elements_of(states)

        # Extract Keplerian elements
        sma = kep.semi_major_axis.m
        ecc = kep.eccentricity
        inc = kep.inclination.radians
        raan = kep.longitude_of_ascending_node.radians
        aop = kep.argument_of_periapsis.radians
        ma = kep.mean_anomaly.radians

        # Return Keplerian elements
        return np.column_stack((sma, ecc, inc, raan, aop, ma))


class Earth(SkyfieldObject):
    name = "earth"


class Moon(SkyfieldObject):
    name = "moon"
