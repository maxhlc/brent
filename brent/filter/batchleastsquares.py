# Third-party imports
import numpy as np

# Orekit imports
import orekit
from org.orekit.estimation.leastsquares import BatchLSEstimator
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
