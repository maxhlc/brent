# Standard imports
from copy import deepcopy

# Third-party imports
import numpy as np
import pandas as pd
import scipy.optimize

# Orekit imports
import orekit
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.utils import ParameterDriver
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer

# Internal imports
from .covariance import CovarianceProvider
from .observations import generate_observations
from brent.propagators import (
    WrappedPropagator,
    OrekitNumericalPropagator,
    ThalassaNumericalPropagator,
    NumericalPropagatorParameters,
)


class OrekitBatchLeastSquares:
    def __init__(
        self,
        dates: np.ndarray | pd.DatetimeIndex,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
        covarianceProvider: CovarianceProvider,
    ):
        # Create decomposer and optimiser
        matrixDecomposer = QRDecomposer(1e-11)
        optimiser = GaussNewtonOptimizer(matrixDecomposer, False)

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
        # Prevent execution
        # TODO: finish implementation
        raise NotImplementedError

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
        dates: np.ndarray | pd.DatetimeIndex,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
        covarianceProvider: CovarianceProvider,
    ) -> None:
        # Store observation dates and states
        self.dates = np.copy(dates)
        self.states = np.copy(states)

        # Store model
        self.model = model
        self.srp_estimate = model.srp_estimate

        # Store covariance provider
        self.covarianceProvider = covarianceProvider

        # TODO: CR/CD estimation

    def estimate(self) -> ThalassaNumericalPropagator:
        # Extract the dates, states, and model
        dates = self.dates
        states = self.states
        model = self.model
        srp_estimate = self.srp_estimate

        # Extract the initial date
        dateInitial = dates[0]

        def fun(xdata, *params):
            # Extract initial state vector
            stateInitial = np.array(params[0:6])

            # Make a copy of the model parameters
            model_ = deepcopy(model)

            # Adjust model
            # TODO: CD estimation
            if srp_estimate:
                model_.cr = params[6]

            # Create propagator
            propagator = ThalassaNumericalPropagator(dateInitial, stateInitial, model_)

            # Propagate states
            states_ = propagator.propagate(dates)

            # Ensure propagator is destroyed before the next run
            # NOTE: this is due to threading issues within THALASSA
            del propagator

            # Return calculated states
            return states_.ravel()

        # Prepare data for fitting
        xdata = dates
        ydata = states.ravel()

        # Calculate residual noise
        # NOTE: scipy.curve_fit expects covariances
        sigma = self.covarianceProvider.covariance(states)

        # Set initial guess
        p0 = states[0, :]
        if self.srp_estimate:
            p0 = np.append(p0, model.cr)

        # Execute optimiser
        popt, pcov = scipy.optimize.curve_fit(
            fun,
            xdata,
            ydata,
            p0,
            sigma=sigma,
            absolute_sigma=True,
        )

        # Store optimisation results
        self.popt = popt
        self.pcov = pcov

        # Extract estimated state
        stateEstimated = popt[0:6]

        # Extract estimated model
        modelEstimated = deepcopy(model)
        if srp_estimate:
            modelEstimated.cr = popt[6]

        # Return solution
        return ThalassaNumericalPropagator(dateInitial, stateEstimated, modelEstimated)

    def covariance(self) -> np.ndarray:
        # Return covariance matrix
        return self.pcov
