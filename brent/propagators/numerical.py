# Standard imports
from dataclasses import dataclass
from datetime import datetime, timedelta
import multiprocessing as mp
import queue

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.drag import DragForce, DragSensitive, IsotropicDrag
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
from org.orekit.models.earth.atmosphere import NRLMSISE00, NRLMSISE00InputParameters
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
    cr: float
    area_srp: float

    # Drag
    drag: bool
    drag_estimate: bool
    cd: float
    area_drag: float


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
            CartesianOrbit(
                state_,
                Constants.DEFAULT_ECI,
                Constants.DEFAULT_MU,
            )
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

        # Create Earth body
        earthBody = OneAxisEllipsoid(
            Constants.DEFAULT_RADIUS,
            Constants.DEFAULT_FLATTENING,
            Constants.DEFAULT_ECEF,
        )

        # Get celestial bodies
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()

        # Create gravity model
        if model.potential:
            # Load geopotential model
            gravityFieldProvider = GravityFieldFactory.getNormalizedProvider(
                model.potential_degree,
                model.potential_order,
            )

            # Create geopotential force model
            gravityForceModel = HolmesFeatherstoneAttractionModel(
                Constants.DEFAULT_ECEF,
                gravityFieldProvider,
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
                earthBody,
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

        # Drag model
        if model.drag:
            # Create drag force
            drag = DragForce(
                NRLMSISE00(NRLMSISE00InputParameters(), sun, earthBody),
                IsotropicDrag(model.area_drag, model.cd),
            )

            # Enable CD estimation
            if model.drag_estimate:
                # Iterate through drivers
                for driver in srp.getParametersDrivers():
                    # Enable drag driver
                    if driver.getName() == DragSensitive.DRAG_COEFFICIENT:
                        driver.setSelected(True)

            # Add drag to force model
            propagatorBuilder.addForceModel(drag)

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

    def __init__(
        self,
        date: datetime,
        state: np.ndarray,
        model: NumericalPropagatorParameters,
    ) -> None:
        # Store initial date and state
        self.date = date
        self.state = state

        # Declare model
        self.model = pythalassa.Model()
        # Set geopotential model
        self.model.insgrav = pythalassa.NONSPHERICAL if model.potential else pythalassa.SPHERICAL
        self.model.gdeg = model.potential_degree
        self.model.gord = model.potential_order
        # Set lunisolar perturbations
        self.model.iephem = pythalassa.EPHEM_DE431  # TODO: switch to JPL?
        self.model.isun = pythalassa.SUN_ENABLED if model.sun else pythalassa.SUN_DISABLED
        self.model.imoon = pythalassa.MOON_ENABLED if model.moon else pythalassa.MOON_DISABLED
        # Set drag model
        self.model.idrag = pythalassa.DRAG_NRLMSISE00 if model.drag else pythalassa.DRAG_DISABLED
        self.model.iF107 = pythalassa.FLUX_VARIABLE
        # Set SRP model
        self.model.iSRP = pythalassa.SRP_ENABLED_CONICAL if model.srp else pythalassa.SRP_DISABLED

        # Declare paths
        self.paths = pythalassa.Paths()
        self.paths.phys_path = brent.paths.THALASSA_DATA_PHYSICAL
        self.paths.earth_path = brent.paths.THALASSA_DATA_EARTH
        self.paths.kernel_path = brent.paths.THALASSA_DATA_KERNEL

        # Declare settings
        # TODO: set
        self.settings = pythalassa.Settings()
        self.settings.eqs = pythalassa.EDROMO_T  # TODO: add options for different methods?
        self.settings.tol = 1e-14

        # Declare spacecraft
        self.spacecraft = pythalassa.Spacecraft()
        # Set mass
        self.spacecraft.mass = model.mass
        # Set drag properties
        self.spacecraft.area_drag = model.area_drag
        self.spacecraft.cd = model.cd
        # Set SRP properties
        self.spacecraft.area_srp = model.area_srp
        self.spacecraft.cr = model.cr

    @staticmethod
    def _propagate(
        results: mp.Queue,
        model: pythalassa.Model,
        paths: pythalassa.Paths,
        settings: pythalassa.Settings,
        spacecraft: pythalassa.Spacecraft,
        dates,
        state,
    ) -> None:
        # Create propagator
        propagator = pythalassa.Propagator(model, paths, settings, spacecraft)

        # Convert state vector to [km] and [km/s]
        state_ = state.ravel() / 1000.0

        # Convert dates to MJD floats
        dates_ = dates_to_MJD(dates)

        # Propagate
        states_ = propagator.propagate(dates_, state_)

        # Transpose to row vectors and convert to [m] and [m/s]
        states = states_.T * 1000.0

        # Put states onto results queue
        results.put(states)

    def propagate(self, dates, frame=Constants.DEFAULT_ECI, timeout: float = 600.0):
        # Check requested frame is compatible with THALASSA
        if frame != FramesFactory.getGCRF():
            # TODO: check correct frame
            raise ValueError("THALASSA only supports GCRF")

        # Ensure dates increase/decrease monotonically
        diff = np.diff(dates)
        monotonic = np.all(diff >= np.timedelta64(0, "D")) or np.all(diff <= np.timedelta64(0, "D"))
        if not monotonic:
            raise ValueError("Dates must be monotonically increasing or decreasing")

        # Check if first date matches stored initial date
        if dates[0] != self.date:
            # Propagate state to initial date
            state = self.propagate(np.array([self.date, dates[0]]))[1, :]
        else:
            # Use stored initial state
            state = self.state

        # Propagate states
        # NOTE: the propagator is executed as a subprocess to avoid issues encountered when
        #       creating and destroying large numbers of propagators using SPICE kernels
        results = mp.Queue()

        # Create propagation process
        process = mp.Process(
            target=ThalassaNumericalPropagator._propagate,
            args=(
                results,
                self.model,
                self.paths,
                self.settings,
                self.spacecraft,
                dates,
                state,
            ),
        )

        # Try to propagate
        try:
            # Start propagation
            process.start()

            # Wait for results
            states = results.get(timeout=timeout)
        except queue.Empty:
            # Raise error due to lack of results
            raise RuntimeError("Propagation timed out")
        finally:
            # Ensure process is terminated
            process.terminate()

        # Return states
        return states
