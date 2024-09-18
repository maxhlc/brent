# Standard imports
from copy import deepcopy

# Third-party imports
import numpy as np
import pandas as pd
import scipy.optimize

# Orekit imports
import orekit
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.utils import ParameterDriver
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer

# Internal imports
from .covariance import CovarianceProvider
from .observations import generate_observations
from brent import Constants
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

        # Store model
        self.model = model

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
        self._estimate = NumericalPropagator.cast_(self.estimator.estimate()[0])

        # Execute fit propagator
        return WrappedPropagator(self._estimate)

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

    def getCovariance(self) -> np.ndarray:
        # Extract estimated covariance
        covariance_ = self.estimator.getPhysicalCovariances(1e-16)

        # Convert to NumPy array
        covariance = np.array(
            [covariance_.getRow(irow) for irow in range(covariance_.getRowDimension())]
        )

        # Return fit covariance
        return covariance

    def getModel(self) -> NumericalPropagatorParameters:
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
        self.drag_estimate = model.drag_estimate

        # Throw error if SRP estimation enabled without SRP
        # TODO: force enable SRP?
        if model.srp_estimate and not model.srp:
            raise ValueError("Cannot estimate SRP if disabled in model")

        # Throw error if drag estimation enabled without drag
        # TODO: force enable drag?
        if model.drag_estimate and not model.drag:
            raise ValueError("Cannot estimate drag if disabled in model")

        # Store covariance provider
        self.covarianceProvider = covarianceProvider

    def estimate(self) -> ThalassaNumericalPropagator:
        # Extract the dates, states, and model
        dates = self.dates
        states = self.states
        model = self.model
        srp_estimate = self.srp_estimate
        drag_estimate = self.drag_estimate

        # Extract the initial date
        dateInitial = dates[0]

        def fun(x):
            # Extract initial state vector
            stateInitial = np.array(x[0:6])

            # Make a copy of the model parameters
            model_ = deepcopy(model)

            # Adjust model
            if srp_estimate and drag_estimate:
                model_.cr = x[6]
                model_.drag = x[7]
            elif srp_estimate:
                model_.cr = x[6]
            elif drag_estimate:
                model_.drag = x[6]

            # Create propagator
            propagator = ThalassaNumericalPropagator(dateInitial, stateInitial, model_)

            # Propagate states
            states_ = propagator.propagate(dates)

            # Return calculated states
            return (states_ - states).ravel()

        # Set initial guess
        x0 = states[0, :]
        if self.srp_estimate:
            x0 = np.append(x0, model.cr)
        if self.drag_estimate:
            x0 = np.append(x0, model.cd)

        # Calculate scaling units
        r0 = np.linalg.norm(states[0, 0:3])
        v0 = np.linalg.norm(states[0, 3:6])
        lu = 1.0 / (2.0 / r0 - v0**2 / Constants.DEFAULT_MU)
        vu = np.sqrt(Constants.DEFAULT_MU / lu)
        x_scale = np.array([lu, lu, lu, vu, vu, vu])
        if self.srp_estimate:
            x_scale = np.append(x_scale, 0.01)
        if self.drag_estimate:
            x_scale = np.append(x_scale, 0.01)

        # Execute optimiser
        # TODO: consider switching to Jacobian scaling?
        sol = scipy.optimize.least_squares(
            fun,
            x0,
            method="lm",
            x_scale=x_scale,
        )

        # Store optimisation results
        # TODO: estimate covariance with weights?
        self.popt = sol.x
        self.pcov = np.linalg.inv(sol.jac.T @ sol.jac)

        # Extract estimated state
        stateEstimated = self.getEstimatedState()

        # Extract estimated model
        modelEstimated = self.getModel()

        # Return solution
        return ThalassaNumericalPropagator(dateInitial, stateEstimated, modelEstimated)

    def getCovariance(self) -> np.ndarray:
        # Return covariance matrix
        return self.pcov

    def getEstimatedState(self) -> np.ndarray:
        # Return estimated state
        return self.popt[0:6]

    def getModel(self) -> NumericalPropagatorParameters:
        # Extract optimisation results
        popt = self.popt

        # Extract estimated model
        model = deepcopy(self.model)
        if self.srp_estimate and self.drag_estimate:
            model.cr = popt[6]
            model.cd = popt[7]
        elif self.srp_estimate:
            model.cr = popt[6]
        elif self.drag_estimate:
            model.drag = popt[6]

        # Return estimated model
        return model
