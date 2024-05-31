# Third-party imports
import numpy as np

# Orekit imports
import orekit
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.utils import ParameterDriver
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer

# Internal imports
import brent.propagators


class BatchLeastSquares:
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

    def estimate(self):
        # Execute fit propagator
        return brent.propagators.WrappedPropagator(self.estimator.estimate()[0])

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
