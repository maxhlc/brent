# Third-party imports
import numpy as np
import scipy.optimize

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.estimation.measurements import ObservableSatellite, PV
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from .covariance import CovarianceProvider
from .weight import Weight
import brent.frames
import brent.propagators


class OrekitBatchLeastSquares:
    def __init__(self, dates, states, model, covarianceProvider=CovarianceProvider()):
        # Generate states in Orekit format
        states_ = [
            TimeStampedPVCoordinates(
                datetime_to_absolutedate(date),
                Vector3D(*state[0:3].tolist()),
                Vector3D(*state[3:6].tolist()),
            )
            for date, state in zip(dates, states)
        ]

        # Create propagator builder
        propagatorBuilder = brent.propagators.default_propagator_builder_(
            states_[0], model
        )

        # Create decomposer and optimiser
        matrixDecomposer = QRDecomposer(1e-11)
        optimiser = GaussNewtonOptimizer(matrixDecomposer, False)

        # Create estimator
        estimator = BatchLSEstimator(optimiser, propagatorBuilder)

        # Set estimator parameters
        estimator.setParametersConvergenceThreshold(1e-3)
        estimator.setMaxIterations(25)
        estimator.setMaxEvaluations(35)

        # Create satellite
        satellite = ObservableSatellite(0)

        # Create observations
        for state, state_ in zip(states, states_):
            # Calculate covariance matrix
            covariance = covarianceProvider(state)

            # Extract position and velocity sigmas
            sigmaPosition = np.sqrt(np.diag(covariance[0:3, 0:3])).tolist()
            sigmaVelocity = np.sqrt(np.diag(covariance[3:6, 3:6])).tolist()

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

            # Add observation to estimator
            estimator.addMeasurement(pv)

        # Store estimator
        self.estimator = estimator

    def estimate(self):
        # Execute fit propagator
        return brent.propagators.Propagator(self.estimator.estimate()[0])

    def covariance(self):
        # Extract estimated covariance
        covariance_ = self.estimator.getPhysicalCovariances(1e-16)

        # Convert to NumPy array
        covariance = np.array(
            [covariance_.getRow(irow) for irow in range(covariance_.getRowDimension())]
        )

        # Return fit covariance
        return covariance


class BatchLeastSquares:
    def __init__(self, func, wfunc=Weight(), eps=1e-8, niter=25, decov=1e-6):
        # Store functions
        self.func = func
        self.wfunc = wfunc

        # Store solver parameters
        self.eps = eps
        self.niter = niter
        self.decov = decov

        # Declare lists for intermediate results
        self.idx = 0
        self.x_ = []
        self.y_ = []
        self.J_ = []
        self.W_ = []
        self.A_ = []
        self.b_ = []
        self.e_ = []
        self.dx_ = []
        self.de_ = []

    def estimate(self, x_):
        # Make a copy of the initial guess
        x = np.copy(x_)

        # Iterate
        for iter in range(self.niter):
            # Calculate error vector and its Jacobian
            y = self.func(x)
            J = scipy.optimize.approx_fprime(x, self.func, self.eps * np.abs(x))

            # Calculate weight matrix
            W = self.wfunc(y)

            # Calculate linear system matrices
            A = J.T @ W @ J
            b = J.T @ W @ y

            # Calculate error
            e = np.sqrt(y.T @ W @ y)

            # Calculate solution update
            dx = -np.linalg.solve(A, b)

            # Store intermediate results
            self.x_.append(x)
            self.y_.append(y)
            self.J_.append(J)
            self.W_.append(W)
            self.A_.append(A)
            self.b_.append(b)
            self.e_.append(e)
            self.dx_.append(dx)

            # Calculate percentage change
            if iter >= 1:
                de = self.e_[iter] / self.e_[iter - 1] - 1
            else:
                de = np.inf

            # Store percentage change
            self.de_.append(de)

            # Break loop if converged
            if np.abs(de) < self.decov:
                break

            # Update solution
            x += dx

        # Calculate optimal iteration
        self.idx = np.argmin(self.e_)

        # Return optimal solution
        return self.x_[self.idx]

    def covariance(self):
        # Return fit covariance
        return np.linalg.inv(self.A_[self.idx])
