# Standard imports
from dataclasses import dataclass
from datetime import datetime

# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.bodies import CelestialBodyFactory
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
from org.orekit.orbits import CartesianOrbit, PositionAngle
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.conversion import (
    NumericalPropagatorBuilder,
    DormandPrince853IntegratorBuilder,
)
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from .constants import DEFAULT_ECI, DEFAULT_ECEF, DEFAULT_MU, DEFAULT_RADIUS

# Default parameters
DEFAULT_INTEGRATOR = DormandPrince853IntegratorBuilder(0.1, 300.0, 1e-3)


@dataclass
class ModelParameters:
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


def default_propagator_builder_(
    state: TimeStampedPVCoordinates, model: ModelParameters
):
    # Create propagator builder
    propagatorBuilder = NumericalPropagatorBuilder(
        CartesianOrbit(state, DEFAULT_ECI, DEFAULT_MU),
        DEFAULT_INTEGRATOR,
        PositionAngle.MEAN,
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
            DEFAULT_ECEF, gravityFieldProvider
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
            DEFAULT_RADIUS,
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


def default_propagator_builder(
    date: datetime, state: np.ndarray, model: ModelParameters
):
    # Convert date and state to Orekit format
    date_ = datetime_to_absolutedate(date)
    pos_ = Vector3D(*state[0:3].tolist())
    vel_ = Vector3D(*state[3:6].tolist())
    state_ = TimeStampedPVCoordinates(date_, pos_, vel_)

    # Return default propagator builder
    return default_propagator_builder_(state_, model)


def default_propagator(date: datetime, state: np.ndarray, model: ModelParameters):
    # Create propagator builder
    propagatorBuilder = default_propagator_builder(date, state, model)

    # Calculate number of parameters
    n = len(propagatorBuilder.getSelectedNormalizedParameters())

    # Create propagator
    propagator = propagatorBuilder.buildPropagator(n * [1.0])

    # Convert date and state to Orekit format
    date_ = datetime_to_absolutedate(date)
    pos_ = Vector3D(*state[0:3].tolist())
    vel_ = Vector3D(*state[3:6].tolist())
    state_ = TimeStampedPVCoordinates(date_, pos_, vel_)
    spacecraftState = SpacecraftState(CartesianOrbit(state_, DEFAULT_ECI, DEFAULT_MU))

    # Ensure initial state is correct
    propagator.setInitialState(spacecraftState)

    # Create default propagator
    return propagator
