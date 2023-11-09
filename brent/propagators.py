# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import absolutedate_to_datetime, datetime_to_absolutedate
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import CartesianOrbit, PositionAngle
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.analytical import AggregateBoundedPropagator
from org.orekit.propagation.analytical.tle import TLEPropagator
from org.orekit.propagation.conversion import (
    NumericalPropagatorBuilder,
    DormandPrince853IntegratorBuilder,
)
from org.orekit.utils import IERSConventions, Constants, TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
import java.util

# Default parameters
DEFAULT_ECI = FramesFactory.getEME2000()
DEFAULT_ECEF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
DEFAULT_MU = Constants.EIGEN5C_EARTH_MU
DEFAULT_INTEGRATOR = DormandPrince853IntegratorBuilder(0.001, 300.0, 1.0)
DEFAULT_DEGREE = 10
DEFAULT_ORDER = 10


def default_propagator_builder_(state):
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


def default_propagator_builder(date, state):
    # Convert date and state to Orekit format
    date = datetime_to_absolutedate(date)
    pos = Vector3D(*state[0:3].tolist())
    vel = Vector3D(*state[3:6].tolist())
    state = TimeStampedPVCoordinates(date, pos, vel)

    # Return default propagator builder
    return default_propagator_builder_(state)


def default_propagator(date, state):
    # Create default propagator
    return default_propagator_builder(date, state).buildPropagator(6 * [1.0])


def tles_to_propagator(tles):
    # Check for number of TLEs
    if len(tles) < 1:
        raise ValueError("Insufficent number of TLEs")

    # TODO: sort TLEs to ensure in order?

    # Declare map for propagators
    propagatorMap = java.util.TreeMap()

    # Iterate through TLEs
    for tle in tles:
        # Extract epoch date
        epoch = tle.getDate()

        # Create propagator
        propagator = TLEPropagator.selectExtrapolator(tle)

        # Add to map
        propagatorMap.put(epoch, propagator)

    # Extract start and end dates
    # TODO: review dates
    dateStart = tles[0].getDate()
    dateEnd = tles[-1].getDate()

    # Return aggregate propagator
    return AggregateBoundedPropagator(propagatorMap, dateStart, dateEnd)


def propagate(propagator, dates, frame=DEFAULT_ECI):
    # Return propagated states
    return [propagator.getPVCoordinates(date, frame) for date in dates]


def pv_to_array(states):
    # Convert dates
    dates_ = [absolutedate_to_datetime(state.getDate()) for state in states]

    # Convert states
    states_ = np.array(
        [
            np.concatenate(
                (state.getPosition().toArray(), state.getVelocity().toArray())
            )
            for state in states
        ]
    )

    # Return dates and states
    return dates_, states_
