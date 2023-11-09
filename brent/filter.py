# Third-party imports
import numpy as np
import scipy.optimize

# Orekit imports
import orekit
from org.orekit.estimation.measurements import ObservableSatellite, PV
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.orekit.estimation.leastsquares import BatchLSEstimator

# Internal imports
import brent.propagators


class OrekitBatchLeastSquares:
    def __init__(self, states):
        # Create propagator builder
        propagatorBuilder = brent.propagators.default_propagator_builder_(states[0])

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
        for state in states:
            # Create observation
            pv = PV(
                state.getDate(),
                state.getPosition(),
                state.getVelocity(),
                1.0,
                1.0,
                1.0,
                satellite,
            )

            # Add observation to estimator
            estimator.addMeasurement(pv)

        # Store estimator
        self.estimator = estimator

    def estimate(self):
        # Execute estimator
        return self.estimator.estimate()[0]


class Weight:
    def __init__(self):
        pass

    def __call__(self, y):
        # Return identity matrix
        return np.identity(len(y))


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

    def estimate(self, x):
        # Iterate
        for iter in range(self.niter):
            # Calculate error vector and its Jacobian
            y = self.func(x)
            J = scipy.optimize.approx_fprime(x, self.func, self.eps * x)

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
