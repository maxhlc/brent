# Standard imports
from copy import deepcopy
from typing import List

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.estimation.measurements import ObservableSatellite, PV
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer

# Internal imports
from .batchleastsquares import BatchLeastSquares
from brent.covariance import Covariance, IdentityCovariance
from brent.propagators import (
    WrappedPropagator,
    OrekitNumericalPropagator,
    NumericalPropagatorParameters,
)


def generate_observations(
    dates,
    states: np.ndarray,
    covarianceProvider: Covariance,
) -> List[PV]:
    # Generate states in Orekit format
    states_ = [
        TimeStampedPVCoordinates(
            datetime_to_absolutedate(date),
            Vector3D(*state[0:3].tolist()),
            Vector3D(*state[3:6].tolist()),
        )
        for date, state in zip(dates, states)
    ]

    # Create satellite
    satellite = ObservableSatellite(0)

    # Declare observation list
    observations = []

    # Create observations
    for state, state_ in zip(states, states_):
        # Calculate covariance matrix
        covariance = covarianceProvider.covariance(state)

        # Calculate diagonal standard deviations
        sigmas = np.sqrt(np.diag(covariance))

        # Extract position and velocity sigmas
        sigmaPosition = sigmas[0:3].tolist()
        sigmaVelocity = sigmas[3:6].tolist()

        # Create observation
        pv = PV(
            state_.getDate(),
            state_.getPosition(),
            state_.getVelocity(),
            sigmaPosition,
            sigmaVelocity,
            1.0,
            satellite,
        )

        # Append to observation list
        observations.append(pv)

    # Return observations
    return observations


class OrekitBatchLeastSquares(BatchLeastSquares):
    def __init__(
        self,
        dates: np.ndarray | pd.DatetimeIndex,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
        covarianceProvider: Covariance,
    ):
        # Create decomposer and optimiser
        matrixDecomposer = QRDecomposer(1e-11)
        optimiser = GaussNewtonOptimizer(matrixDecomposer, False)

        # Store model
        self.model = model

        # Throw error if SRP estimation enabled without SRP
        # TODO: force enable SRP?
        if model.srp_estimate and not model.srp:
            raise ValueError("Cannot estimate SRP if disabled in model")

        # Throw error if drag estimation enabled without drag
        # TODO: force enable drag?
        if model.drag_estimate and not model.drag:
            raise ValueError("Cannot estimate drag if disabled in model")

        # Create builder
        builder = OrekitNumericalPropagator.builder(dates[0], states[0, :], model)

        # Create estimator
        estimator = BatchLSEstimator(optimiser, builder)

        # Set estimator parameters
        estimator.setParametersConvergenceThreshold(1e-3)
        estimator.setMaxIterations(25)
        estimator.setMaxEvaluations(35)

        # Generate observations
        observations = generate_observations(dates, states, covarianceProvider)

        # Add observations to estimator
        for observation in observations:
            estimator.addMeasurement(observation)

        # Store estimator
        self.estimator = estimator

    def estimate(self) -> WrappedPropagator:
        # Generate estimate
        self._estimate: NumericalPropagator
        self._estimate = NumericalPropagator.cast_(self.estimator.estimate()[0])

        # Execute fit propagator
        return WrappedPropagator(self._estimate)

    def getEstimatedState(self) -> np.ndarray:
        # Extract estimated state
        state_ = self._estimate.getInitialState()

        # Convert to NumPy vector
        # TODO: specify frame?
        pv = state_.getPVCoordinates()
        pos = pv.getPosition().toArray()
        vel = pv.getVelocity().toArray()
        state = np.array(pos + vel)

        # Return estimated state vector
        return state

    def getEstimatedCovariance(self) -> np.ndarray:
        # Extract estimated covariance
        covariance_ = self.estimator.getPhysicalCovariances(1e-16)

        # Convert to NumPy array
        covariance = np.array(
            [covariance_.getRow(irow) for irow in range(covariance_.getRowDimension())]
        )

        # Return fit covariance
        return covariance

    def getEstimatedModel(self) -> NumericalPropagatorParameters:
        # Extract estimated model
        model = deepcopy(self.model)

        # Extract force models
        forceModels = self._estimate.getAllForceModels()

        # TODO: make much cleaner

        # Iterate through force models
        for idx in range(forceModels.size()):
            # Extract force model
            forceModel = forceModels.get(idx)

            # Extract force model drivers
            drivers = forceModel.getParametersDrivers()

            # Iterate through drivers
            for jdx in range(drivers.size()):
                # Extract driver
                driver = drivers.get(jdx)

                # Extract driver name and value
                driverName = driver.getName()
                driverValue = driver.getValue()

                # Update CR
                if driverName == "reflection coefficient":
                    model.cr = driverValue

        # Return estimated model
        return model
