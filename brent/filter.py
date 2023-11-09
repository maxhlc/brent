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
