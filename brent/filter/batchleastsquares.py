# Third-party imports
import numpy as np

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
import brent.frames
import brent.propagators


class BatchLeastSquares:
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
