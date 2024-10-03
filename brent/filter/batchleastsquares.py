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
        self.dates = pd.to_datetime(np.copy(dates))
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

    @staticmethod
    def _residuals(
        params: np.ndarray,
        dates: pd.DatetimeIndex,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
    ) -> np.ndarray:
        # Calculate states
        states_ = ThalassaBatchLeastSquares._propagate(params, dates, states, model)

        # Calculate residuals
        residuals = (states_ - states).ravel()

        # Return residuals
        return residuals

    @staticmethod
    def _propagate(
        params: np.ndarray,
        dates: pd.DatetimeIndex,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
    ) -> np.ndarray:
        # Extract initial state vector
        state = params[0:6]

        # Make a copy of the model parameters
        model_ = deepcopy(model)

        # Adjust model
        if model_.srp_estimate and model_.drag_estimate:
            model_.cr = params[6]
            model_.drag = params[7]
        elif model_.srp_estimate:
            model_.cr = params[6]
        elif model_.drag_estimate:
            model_.drag = params[6]

        # Create propagator
        propagator = ThalassaNumericalPropagator(dates[0], state, model_)

        # Propagate states
        states = propagator.propagate(dates)

        # Return propagated states
        return states

    def estimate(self) -> ThalassaNumericalPropagator:
        # Extract the dates, states, and model
        dates = self.dates
        states = self.states
        model = self.model

        # Set initial guess
        p0 = states[0, :]
        if self.srp_estimate:
            p0 = np.append(p0, model.cr)
        if self.drag_estimate:
            p0 = np.append(p0, model.cd)

        # Calculate observation covariance
        # NOTE: only using the diagonal terms, matching generate_observations
        covariance = self.covarianceProvider.covariance(states)
        covarianceDiagonal = np.diag(np.diag(covariance))

        # Define function
        def fun(_, *params):
            # Calculate states
            states_ = ThalassaBatchLeastSquares._propagate(
                params=np.array(params),
                dates=dates,
                states=states,
                model=model,
            )

            # Return column vector of states
            return states_.ravel()

        # Execute optimiser
        x = dates
        y = states.ravel()
        popt, pcov = scipy.optimize.curve_fit(
            fun,
            x,  # Not used by function
            y,
            p0,
            sigma=covarianceDiagonal,
            absolute_sigma=True,
            method="lm",
        )

        # Store optimisation results
        self.popt = popt
        self.pcov = pcov

        # Extract estimated state
        stateEstimated = self.getEstimatedState()

        # Extract estimated model
        modelEstimated = self.getModel()

        # Return solution
        return ThalassaNumericalPropagator(dates[0], stateEstimated, modelEstimated)

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
