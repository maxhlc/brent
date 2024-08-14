# Standard imports
from copy import deepcopy

# Third-party imports
import numpy as np
import scipy.optimize

# Orekit imports
import orekit
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.utils import ParameterDriver
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer

# Internal imports
from brent import Constants
from brent.propagators import (
    WrappedPropagator,
    ThalassaNumericalPropagator,
    NumericalPropagatorParameters,
)


class OrekitBatchLeastSquares:
    def __init__(self, builder, observations):
        # Create decomposer and optimiser
        matrixDecomposer = QRDecomposer(1e-11)
        optimiser = GaussNewtonOptimizer(matrixDecomposer, False)

        # Create estimator
        estimator = BatchLSEstimator(optimiser, builder)

        # Set estimator parameters
        estimator.setParametersConvergenceThreshold(1e-3)
        estimator.setMaxIterations(25)
        estimator.setMaxEvaluations(35)

        # Add observations to estimator
        for observation in observations:
            estimator.addMeasurement(observation)

        # Store estimator
        self.estimator = estimator

    def estimate(self) -> WrappedPropagator:
        # Execute fit propagator
        return WrappedPropagator(self.estimator.estimate()[0])

    def _jacobian(self) -> np.ndarray:
        # Extract weighted Jacobian matrix from the estimator
        jacobian_ = self.estimator.getOptimum().getJacobian()

        # Return Jacobian matrix
        return np.array(
            [jacobian_.getRow(irow) for irow in range(jacobian_.getRowDimension())]
        )

    def _scale_factors(self) -> np.ndarray:
        # Extract parameter drivers
        # TODO: account for additional drivers (e.g. SRP estimation, etc.)
        drivers_ = self.estimator.getOrbitalParametersDrivers(False)
        drivers = [
            ParameterDriver.cast_(drivers_.getDrivers().get(idx))
            for idx in range(drivers_.getNbParams())
        ]

        # Extract scaling factors
        # NOTE: assumes order is the same as in the Jacobian
        scales = np.array([driver.getScale() for driver in drivers])

        # Return scaling factors
        return scales

    def jacobian(self) -> np.ndarray:
        # Extract weighted Jacobian matrix
        # TODO: figure out how to remove the weighting!
        jacobian = self._jacobian()

        # Extract scaling factors
        scales = self._scale_factors()
        scaling_matrix = scales.reshape((1, -1)) / scales.reshape((-1, 1))

        # Scale Jacobian matrix to physical dimensions
        n = jacobian.shape[0] // scaling_matrix.shape[0]
        scaling_matrix_tiled = np.tile(scaling_matrix, (n, 1))
        jacobian /= scaling_matrix_tiled

        # Return Jacobian matrix
        return jacobian

    def covariance(self) -> np.ndarray:
        # Extract estimated covariance
        covariance_ = self.estimator.getPhysicalCovariances(1e-16)

        # Convert to NumPy array
        covariance = np.array(
            [covariance_.getRow(irow) for irow in range(covariance_.getRowDimension())]
        )

        # Return fit covariance
        return covariance


class ThalassaBatchLeastSquares:

    def __init__(
        self,
        dates: np.ndarray,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
    ) -> None:
        # Store observation dates and states
        self.dates = np.copy(dates)
        self.states = np.copy(states)

        # Store model
        self.model = model

        # TODO: observation uncertainties
        # TODO: CR/CD estimation

    def estimate(self) -> ThalassaNumericalPropagator:
        # Extract the dates, states, and model
        dates = self.dates
        states = self.states
        model = self.model

        # Extract the initial date
        dateInitial = dates[0]

        # Set initial guess
        x0 = states[0, :]

        # Calculate position and velocity magnitudes
        r0 = np.linalg.norm(states[0, 0:3])
        v0 = np.linalg.norm(states[0, 3:6])

        # Calculate scaling units
        lu = 1.0 / (2.0 / r0 - v0**2 / Constants.DEFAULT_MU)
        vu = np.sqrt(Constants.DEFAULT_MU / lu)
        fscale = np.array([[lu, lu, lu, vu, vu, vu]])

        def fun(x):
            # Extract initial state vector
            stateInitial = x[0:6]

            # Make a copy of the model parameters
            model_ = deepcopy(model)

            # TODO: adjust model if estimating CR/CD

            # Create propagator
            propagator = ThalassaNumericalPropagator(dateInitial, stateInitial, model_)

            # Propagate states
            states_ = propagator.propagate(dates)

            # Ensure propagator is destroyed before the next run
            # NOTE: this is due to threading issues within THALASSA
            del propagator

            # Calculate state errors
            delta = states_ - states

            # Scale state error
            delta /= fscale

            # Return vector of state errors
            return delta.ravel()

        # Execute optimiser
        sol = scipy.optimize.least_squares(
            fun,
            x0,
            x_scale=fscale.ravel(),
        )

        # Extract estimated state
        stateEstimated = sol.x[0:6]

        # Return solution
        # TODO: return with updated numerical model
        return ThalassaNumericalPropagator(dateInitial, stateEstimated, model)
