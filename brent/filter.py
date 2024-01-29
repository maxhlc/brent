# Third-party imports
import numpy as np
import scipy.optimize

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.estimation.measurements import ObservableSatellite, PV
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.utils import PVCoordinates, TimeStampedPVCoordinates
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.hipparchus.geometry.euclidean.threed import Vector3D
import java.util

# Internal imports
import brent.frames
import brent.propagators


class CovarianceProvider:
    def __init__(self):
        pass

    def __call__(self, state):
        # Return identity matrix
        return np.identity(len(state))


class RTNCovarianceProvider(CovarianceProvider):
    def __init__(self, sigma):
        # Store standard deviations
        self.sigma = sigma

        # Create RTN covariance matrix
        self.covarianceRTN = np.diag(sigma) ** 2

    def __call__(self, state):
        # Calculate RTN transform
        rtn = brent.frames.rtn(state.reshape((1, 6)))[0, :, :]

        # Rotate covariance matrix to inertial frame
        covarianceXYZ = rtn.T @ self.covarianceRTN @ rtn

        # Return inertial covariance
        return covarianceXYZ


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
        propagatorBuilder = brent.propagators.default_propagator_builder_(states_[0], model)

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


class Weight:
    def __init__(self):
        pass

    def __call__(self, y):
        # Return identity matrix
        return np.identity(len(y))


class ConstantWeight(Weight):
    def __init__(self, nvar, weights):
        # Check for consistency
        assert nvar == len(weights.ravel())

        # Store number of variables and weights
        self.nvar = nvar
        self.weights = weights.ravel()

    def __call__(self, y):
        # Calculate number of residuals
        ny = len(y)
        nvar = self.nvar
        nsam = ny // nvar

        # Declare weight matrix
        W = np.identity(len(y))

        # Update weight matrix
        W[np.diag_indices_from(W)] = np.tile(self.weights, nsam)

        # Return weight matrix
        return W


class SampledWeight(Weight):
    def __init__(self, nvar):
        # Store number of variables
        self.nvar = nvar

    def __call__(self, y):
        # Calculate number of residuals
        ny = len(y)
        nvar = self.nvar
        nsam = ny // nvar

        # Declare weight matrix
        W = np.identity(len(y))

        # Calculate residual covariance matrix and extract diagonal terms
        yCov = np.cov(y.reshape((-1, self.nvar)), rowvar=False)
        ySig = np.sqrt(np.diag(yCov))

        # Update weight matrix
        W[np.diag_indices_from(W)] /= np.tile(ySig, nsam)

        # Return weight matrix
        return W


class CovarianceProviderWeight(Weight):
    def __init__(self, states, covarianceProvider):
        # Store number of variables
        self.states = states

        # Extract number of states and variables
        self.nstates = states.shape[0]
        self.nvar = states.shape[1]

        # Calculate number of elements in residual vector
        self.nelem = self.nstates * self.nvar

        # Calculate variances
        self.variances = np.concatenate(
            [np.diag(covarianceProvider(state)) for state in states]
        )

        # Calculate weight matrix
        self.W = np.diag(1.0 / np.sqrt(self.variances))

    def __call__(self, y):
        # Check number of residuals
        assert len(y) == self.nelem

        # Return weight matrix
        return self.W


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
