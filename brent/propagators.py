# Orekit imports
import orekit
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, PositionAngle
from org.orekit.propagation.conversion import (
    NumericalPropagatorBuilder,
    DormandPrince853IntegratorBuilder,
)
from org.orekit.utils import IERSConventions, Constants

# Default parameters
DEFAULT_ECI = FramesFactory.getEME2000()
DEFAULT_ECEF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
DEFAULT_MU = Constants.EIGEN5C_EARTH_MU
DEFAULT_INTEGRATOR = DormandPrince853IntegratorBuilder(0.001, 300.0, 1.0)
DEFAULT_DEGREE = 10
DEFAULT_ORDER = 10


def default_propagator_builder(state):
    # Create propagator builder
    propagatorBuilder = NumericalPropagatorBuilder(
        CartesianOrbit(state, DEFAULT_ECI, DEFAULT_MU),
        DEFAULT_INTEGRATOR,
        PositionAngle.MEAN,
        1.0,
    )

    # Get celestial bodies
    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()

    # Create gravity model
    gravityFieldProvider = GravityFieldFactory.getNormalizedProvider(
        DEFAULT_DEGREE, DEFAULT_ORDER
    )
    gravityForceModel = HolmesFeatherstoneAttractionModel(
        DEFAULT_ECEF, gravityFieldProvider
    )
    propagatorBuilder.addForceModel(gravityForceModel)

    # Moon model
    moonAttraction = ThirdBodyAttraction(moon)
    propagatorBuilder.addForceModel(moonAttraction)

    # Sun model
    sunAttraction = ThirdBodyAttraction(sun)
    propagatorBuilder.addForceModel(sunAttraction)

    # Return propagator builder
    return propagatorBuilder
