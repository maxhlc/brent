# Future imports
from __future__ import annotations

# Standard imports
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict

# Third-party imports
import numpy as np

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
from org.orekit.orbits import CartesianOrbit, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.conversion import (
    NumericalPropagatorBuilder,
    DormandPrince853IntegratorBuilder,
)
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from .propagator import WrappedPropagator
from brent import Constants
from brent.bias import BiasModel, deserialise_bias
from brent.noise import CovarianceProvider, deserialise_noise

# Default parameters
DEFAULT_INTEGRATOR = DormandPrince853IntegratorBuilder(0.1, 300.0, 1e-3)


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

    def serialise(self) -> Dict[str, Any]:
        # Return serialised object
        return asdict(self)

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> NumericalPropagatorParameters:
        # Return deserialised object
        return NumericalPropagatorParameters(**struct)


class NumericalPropagator(WrappedPropagator):
    # Set metadata
    type: str = "numerical"

    def __init__(
        self,
        date: datetime,
        state: np.ndarray,
        model: NumericalPropagatorParameters,
        bias: BiasModel = BiasModel(),
        noise: CovarianceProvider = CovarianceProvider(),
    ):
        # Store parameters
        self.date = date
        self.state = state
        self.model = model

        # Create propagator builder
        propagatorBuilder = NumericalPropagator.builder(date, state, model)

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
        super().__init__(propagator, bias, noise)

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
        return NumericalPropagator.__builder(state_, model)

    def serialise_parameters(self) -> Dict[str, Any]:
        # Return serialised parameters
        return {
            "date": self.date.isoformat(),
            "state": self.state.tolist(),
            "model": self.model.serialise(),
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> NumericalPropagator:
        # Deserialise bias and noise
        bias = deserialise_bias(struct["bias"])
        noise = deserialise_noise(struct["noise"])

        # Deserialise initial date and state
        date = datetime.fromisoformat(struct["parameters"]["date"])
        state = np.array(struct["parameters"]["state"])

        # Deserialise model
        model = NumericalPropagatorParameters.deserialise(struct["parameters"]["model"])

        # Return deserialised model
        return NumericalPropagator(date, state, model, bias, noise)
