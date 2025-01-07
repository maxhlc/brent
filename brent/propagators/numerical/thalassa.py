# Standard imports
from datetime import datetime, timedelta
import multiprocessing as mp
import queue

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from org.orekit.frames import FramesFactory

# External imports
import pythalassa

# Internal imports
from .parameters import NumericalPropagatorParameters
from brent import Constants
import brent.paths
from brent.propagators import Propagator


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
        self.model.insgrav = (
            pythalassa.NONSPHERICAL if model.potential else pythalassa.SPHERICAL
        )
        self.model.gdeg = model.potential_degree
        self.model.gord = model.potential_order
        # Set lunisolar perturbations
        self.model.iephem = pythalassa.EPHEM_DE431
        self.model.isun = (
            pythalassa.SUN_ENABLED if model.sun else pythalassa.SUN_DISABLED
        )
        self.model.imoon = (
            pythalassa.MOON_ENABLED if model.moon else pythalassa.MOON_DISABLED
        )
        # Set drag model
        self.model.idrag = (
            pythalassa.DRAG_NRLMSISE00 if model.drag else pythalassa.DRAG_DISABLED
        )
        self.model.iF107 = pythalassa.FLUX_VARIABLE
        # Set SRP model
        self.model.iSRP = (
            pythalassa.SRP_ENABLED_CONICAL if model.srp else pythalassa.SRP_DISABLED
        )

        # Declare paths
        self.paths = pythalassa.Paths()
        self.paths.phys_path = brent.paths.THALASSA_DATA_PHYSICAL
        self.paths.earth_path = brent.paths.THALASSA_DATA_EARTH
        self.paths.kernel_path = brent.paths.THALASSA_DATA_KERNEL
        self.paths.eop_path = brent.paths.THALASSA_DATA_EOP

        # Declare settings
        # TODO: set
        self.settings = pythalassa.Settings()
        self.settings.eqs = (
            pythalassa.EDROMO_C
        )  # TODO: add options for different methods?
        self.settings.tol = 1e-10

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
    def _static_propagate(
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

    def _get_process(
        self,
        results: mp.Queue,
        dates: pd.DatetimeIndex,
        state: np.ndarray,
    ) -> mp.Process:
        # Return propagation process
        return mp.Process(
            target=ThalassaNumericalPropagator._static_propagate,
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

    @staticmethod
    def _run_process(
        results: mp.Queue,
        process: mp.Process,
        timeout: float,
    ) -> np.ndarray:
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

        # Return propagated states
        return states

    def propagate(self, dates, frame=Constants.DEFAULT_ECI, timeout: float = 600.0):
        # NOTE: the propagator is executed as a subprocess to avoid issues encountered when
        #       creating and destroying large numbers of propagators using SPICE kernels

        # Check requested frame is compatible with THALASSA
        if frame != FramesFactory.getGCRF():
            # TODO: check correct frame
            raise ValueError("THALASSA only supports GCRF")

        # Ensure dates increase/decrease monotonically
        diff = np.diff(dates)
        monotonic = np.all(
            diff >= np.timedelta64(0, "D"),
        ) or np.all(
            diff <= np.timedelta64(0, "D"),
        )
        if not monotonic:
            raise ValueError("Dates must be monotonically increasing or decreasing")

        # Check if first date matches stored initial date
        if dates[0] != self.date:
            # Propagate state to initial date
            state = self.propagate(np.array([self.date, dates[0]]))[1, :]
        else:
            # Use stored initial state
            state = self.state

        # Create results queue
        results = mp.Queue()

        # Create propagation process
        process = self._get_process(results, dates, state)

        # Propagate
        states = ThalassaNumericalPropagator._run_process(results, process, timeout)

        # Return states
        return states
