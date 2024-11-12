# Standard imports
from abc import ABC, abstractmethod
from copy import deepcopy
import multiprocessing as mp
from typing import List

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
from brent.constants import Constants
from brent.frames import Keplerian
from brent.propagators import (
    Propagator,
    WrappedPropagator,
    OrekitNumericalPropagator,
    ThalassaNumericalPropagator,
    NumericalPropagatorParameters,
)


class BatchLeastSquares(ABC):

    @abstractmethod
    def estimate(self) -> Propagator: ...

    @abstractmethod
    def getEstimatedState(self) -> np.ndarray: ...

    @abstractmethod
    def getEstimatedCovariance(self) -> np.ndarray: ...

    @abstractmethod
    def getEstimatedModel(self) -> NumericalPropagatorParameters: ...


class OrekitBatchLeastSquares(BatchLeastSquares):
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


class ThalassaBatchLeastSquares(BatchLeastSquares):

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

        # Calculate scaling units
        keplerian = Keplerian.from_cartesian(dates[0:1], states[0:1, :])
        mu = Constants.DEFAULT_MU
        lu = keplerian[0, 0]
        vu = np.sqrt(mu / lu)

        # Calculate parameter scaling vector
        pscale = np.array(3 * [lu] + 3 * [vu])
        if self.srp_estimate:
            pscale = np.append(pscale, 1e-6)
        if self.drag_estimate:
            pscale = np.append(pscale, 1.0)  # TODO: update

        # Calculate residual scaling vector
        rscale = pscale[0:6]

        # Set initial guess
        # TODO: ratios (e.g. CR * A / m) instead of the coefficient alone
        p0 = states[0, :]
        if self.srp_estimate:
            p0 = np.append(p0, model.cr)
        if self.drag_estimate:
            p0 = np.append(p0, model.cd)
        p0 /= pscale

        # Calculate observation covariance
        covariance = self.covarianceProvider.covariance(states)

        # Extract diagonal covariance matrix (and scale)
        # NOTE: only using the diagonal terms, matching generate_observations
        covarianceDiagonal = np.diag(np.diag(covariance))
        idx = np.diag_indices_from(covarianceDiagonal)
        covarianceDiagonal[idx] /= np.tile(rscale**2, len(dates))

        # Define predictor function
        def fun(_, *p):
            # Scale parameters (non-dimensional to dimensional)
            params = np.array(p) * pscale

            # Calculate states
            states_ = ThalassaBatchLeastSquares._propagate(
                params=params,
                dates=dates,
                states=states,
                model=model,
            )

            # Scale states (dimensional to non-dimensional)
            states_ /= rscale.reshape((1, -1))

            # Return column vector of states
            return states_.ravel()

        # Define queue-based predictor function
        def fun_queue(p: np.ndarray, queue: mp.Queue) -> None:
            states = fun(None, *p)
            queue.put(states)

        # Define parallel predictor function
        # TODO: it would probably be more efficient to run multiple fits in parallel,
        #       instead of Jacobian evaluations in parallel due to the overhead of
        #       creating the subprocesses and the overlap in execution times
        def fun_parallel(ps: np.ndarray) -> List[np.ndarray]:
            # Calculate number of permutations
            nps = ps.shape[0]

            # Declare results queues
            queues = nps * [mp.Queue()]

            # Create processes
            processes = [
                mp.Process(
                    target=fun_queue,
                    args=(p, queue),
                )
                for p, queue in zip(ps, queues)
            ]

            # Start processes
            [process.start() for process in processes]

            # Wait for results
            # TODO: add timeout
            results = [queue.get() for queue in queues]

            # Close processes
            [process.terminate() for process in processes]

            # Return results
            return results

        # Define Jacobian function
        def jac(
            _,
            *p,
            eps: float = 2e-16,
            parallel: bool = False,
        ) -> np.ndarray:
            # Calculate perturbation vector
            x = np.array(p)
            dx = np.sqrt(eps) * np.abs(x)
            dx[dx < 1e-16] = 1e-16

            # Generate perturbations matrix
            x_ = x + np.diag(dx)

            # Create overall input matrix
            xs = np.vstack((x, x_))

            # Calculate reference and perturbed results
            if parallel:
                rs = fun_parallel(xs)
            else:
                rs = [fun(_, *ix) for ix in xs]

            # Split reference and perturbed results
            r0 = rs[0].reshape((-1, 1))
            rp = np.column_stack(rs[1:])

            # Calculate first-order finite difference
            dfdx = (rp - r0) / dx.reshape((1, -1))

            # Return Jacobian matrix
            return dfdx

        # Extract (and scale) observations
        x = dates
        y = states / rscale.reshape((1, -1))
        y = y.ravel()

        # Execute optimiser
        popt, pcov = scipy.optimize.curve_fit(
            fun,
            x,  # Not used by function
            y,
            p0,
            jac=jac,
            sigma=covarianceDiagonal,
            absolute_sigma=True,
            method="lm",
        )

        # Store optimisation scaling and results
        self.pscale = pscale
        self.rscale = rscale
        self.popt = popt
        self.pcov = pcov

        # Extract estimated state
        stateEstimated = self.getEstimatedState()

        # Extract estimated model
        modelEstimated = self.getEstimatedModel()

        # Return solution
        return ThalassaNumericalPropagator(dates[0], stateEstimated, modelEstimated)

    def getEstimatedCovariance(self) -> np.ndarray:
        # Extract optimisation results
        pcov = self.pcov
        pscale = self.pscale

        # Generate scaling matrix
        pscaleMatrix = pscale.reshape((1, -1)) * pscale.reshape((-1, 1))

        # Return covariance matrix
        return pcov * pscaleMatrix

    def getEstimatedState(self) -> np.ndarray:
        # Extract optimisation results
        popt = self.popt
        pscale = self.pscale

        # Return estimated state
        return popt[0:6] * pscale[0:6]

    def getEstimatedModel(self) -> NumericalPropagatorParameters:
        # Extract optimisation results
        popt = self.popt
        pscale = self.pscale

        # Extract estimated model
        # TODO: cleaner solution
        model = deepcopy(self.model)
        if self.srp_estimate and self.drag_estimate:
            model.cr = popt[6] * pscale[6]
            model.cd = popt[7] * pscale[7]
        elif self.srp_estimate:
            model.cr = popt[6] * pscale[6]
        elif self.drag_estimate:
            model.drag = popt[6] * pscale[6]

        # Return estimated model
        return model
