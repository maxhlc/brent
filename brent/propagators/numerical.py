# Standard imports
from dataclasses import dataclass
from datetime import datetime, timedelta

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import (
    SolarRadiationPressure,
    IsotropicRadiationSingleCoefficient,
    RadiationSensitive,
)
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.conversion import (
    NumericalPropagatorBuilder,
    DormandPrince853IntegratorBuilder,
)
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# External imports
import pythalassa

# Internal imports
from .propagator import Propagator, WrappedPropagator
from brent import Constants
import brent.paths

# Default parameters
DEFAULT_INTEGRATOR = DormandPrince853IntegratorBuilder(0.1, 300.0, 1e-8)


@dataclass
class NumericalPropagatorParameters:
    # Physical properties
    mass: float
    area_srp: float
    cr: float

    # Geopotential perturbations
    potential: bool
    potential_degree: int
    potential_order: int

    # Third-body perturbations
    moon: bool
    sun: bool

    # SRP
    srp: bool
    srp_estimate: bool


class OrekitNumericalPropagator(WrappedPropagator):

    def __init__(
        self,
        date: datetime,
        state: np.ndarray,
        model: NumericalPropagatorParameters,
    ):
        # Create propagator builder
        propagatorBuilder = OrekitNumericalPropagator.builder(date, state, model)

        # Calculate number of parameters
        n = len(propagatorBuilder.getSelectedNormalizedParameters())

        # Create propagator
        propagator = propagatorBuilder.buildPropagator(n * [1.0])

        # Convert date and state to Orekit format
        date_ = datetime_to_absolutedate(date)
        pos_ = Vector3D(*state[0:3].tolist())
        vel_ = Vector3D(*state[3:6].tolist())
        state_ = TimeStampedPVCoordinates(date_, pos_, vel_)
        spacecraftState = SpacecraftState(
            CartesianOrbit(state_, Constants.DEFAULT_ECI, Constants.DEFAULT_MU)
        )

        # Ensure initial state is correct
        propagator.setInitialState(spacecraftState)

        # Initialise wrapped propagator
        super().__init__(propagator)

    @staticmethod
    def __builder(
        state: TimeStampedPVCoordinates,
        model: NumericalPropagatorParameters,
    ):
        # Create propagator builder
        propagatorBuilder = NumericalPropagatorBuilder(
            CartesianOrbit(state, Constants.DEFAULT_ECI, Constants.DEFAULT_MU),
            DEFAULT_INTEGRATOR,
            PositionAngleType.MEAN,
            1.0,
        )

        # Set object mass
        propagatorBuilder.setMass(model.mass)

        # Get celestial bodies
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()

        # Create gravity model
        if model.potential:
            # Load geopotential model
            gravityFieldProvider = GravityFieldFactory.getNormalizedProvider(
                model.potential_degree, model.potential_order
            )

            # Create geopotential force model
            gravityForceModel = HolmesFeatherstoneAttractionModel(
                Constants.DEFAULT_ECEF, gravityFieldProvider
            )

            # Add geopotential force model to propagator
            propagatorBuilder.addForceModel(gravityForceModel)

        # Moon model
        if model.moon:
            # Create lunar perturbation
            moonAttraction = ThirdBodyAttraction(moon)

            # Add lunar perturbation to propagator
            propagatorBuilder.addForceModel(moonAttraction)

        # Sun model
        if model.sun:
            # Create solar perturbation
            sunAttraction = ThirdBodyAttraction(sun)

            # Add solar perturbation to propagator
            propagatorBuilder.addForceModel(sunAttraction)

        # SRP model
        if model.srp:
            # Create SRP force
            srp = SolarRadiationPressure(
                sun,
                OneAxisEllipsoid(
                    Constants.DEFAULT_RADIUS,
                    Constants.DEFAULT_FLATTENING,
                    Constants.DEFAULT_ECEF,
                ),
                IsotropicRadiationSingleCoefficient(model.area_srp, model.cr),
            )

            # Enable CR estimation
            if model.srp_estimate:
                # Iterate through drivers
                for driver in srp.getParametersDrivers():
                    # Enable reflection coefficent driver
                    if driver.getName() == RadiationSensitive.REFLECTION_COEFFICIENT:
                        driver.setSelected(True)

            # Add SRP to force model
            propagatorBuilder.addForceModel(srp)

        # Return propagator builder
        return propagatorBuilder

    @staticmethod
    def builder(
        date: datetime,
        state: np.ndarray,
        model: NumericalPropagatorParameters,
    ):
        # Convert date and state to Orekit format
        date_ = datetime_to_absolutedate(date)
        pos_ = Vector3D(*state[0:3].tolist())
        vel_ = Vector3D(*state[3:6].tolist())
        state_ = TimeStampedPVCoordinates(date_, pos_, vel_)

        # Return default propagator builder
        return OrekitNumericalPropagator.__builder(state_, model)


def date_to_MJD(date: datetime | np.datetime64) -> float:
    # Ensure correct date type
    if not isinstance(date, datetime):
        date = pd.to_datetime(date)

    # Calculate delta from 1st January 2000
    delta_J2000 = date - datetime(2000, 1, 1, 0, 0, 0, 0)

    # Convert to float
    delta_J2000_days = delta_J2000 / timedelta(days=1)

    # Add MJD offset
    mjd = delta_J2000_days + 51544.0

    # Return MJD
    return mjd


dates_to_MJD = np.vectorize(date_to_MJD)


class ThalassaNumericalPropagator(Propagator):

    def __init__(self, model_: NumericalPropagatorParameters):
        # Declare model
        model = pythalassa.Model()
        # Set geopotential model
        model.insgrav = pythalassa.NONSPHERICAL if model_.potential else pythalassa.SPHERICAL
        model.gdeg = model_.potential_degree
        model.gord = model_.potential_order
        # Set lunisolar perturbations
        model.iephem = pythalassa.EPHEM_SIMPLE  # TODO: switch to JPL?
        model.isun = pythalassa.SUN_ENABLED if model_.sun else pythalassa.SUN_DISABLED
        model.imoon = pythalassa.MOON_ENABLED if model_.moon else pythalassa.MOON_DISABLED
        # Set drag model
        model.idrag = pythalassa.DRAG_DISABLED  # TODO: setup
        model.iF107 = pythalassa.FLUX_VARIABLE
        # Set SRP model
        model.iSRP = pythalassa.SRP_ENABLED_CONICAL if model_.srp else pythalassa.SRP_DISABLED

        # Declare paths
        paths = pythalassa.Paths()
        paths.phys_path = brent.paths.THALASSA_DATA_PHYSICAL
        paths.earth_path = brent.paths.THALASSA_DATA_EARTH
        paths.kernel_path = brent.paths.THALASSA_DATA_KERNEL

        # Declare settings
        # TODO: set
        settings = pythalassa.Settings()
        settings.eqs = pythalassa.EDROMO_T  # TODO: add options for different methods?
        settings.tol = 1e-10

        # Declare spacecraft
        spacecraft = pythalassa.Spacecraft()
        # Set mass
        spacecraft.mass = model_.mass
        # Set drag properties
        spacecraft.area_drag = 0.0  # TODO: set
        spacecraft.cd = 0.0  # TODO: set
        # Set SRP properties
        spacecraft.area_srp = model_.area_srp
        spacecraft.cr = model_.cr

        # Declare model
        self.propagator = pythalassa.Propagator(model, paths, settings, spacecraft)

    def propagate(self, dates, state, frame=Constants.DEFAULT_ECI):
        # Check requested frame is compatible with THALASSA
        if frame != FramesFactory.getEME2000():
            # TODO: check correct frame
            raise ValueError("THALASSA only supports the EME2000 frame")

        # Convert state vector to [km] and [km/s]
        state_ = state.ravel() / 1000.0

        # Convert dates to MJD floats
        dates_ = dates_to_MJD(dates)

        # Propagate states
        states_ = self.propagator.propagate(dates_, state_)

        # Transpose to row vectors and convert to [m] and [m/s]
        states = states_.T * 1000.0

        # Return states
        return states
