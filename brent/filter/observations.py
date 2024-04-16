# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.estimation.measurements import ObservableSatellite, PV
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D

# Internal imports
from .covariance import CovarianceProvider


def generate_observations(dates, states, covarianceProvider=CovarianceProvider()):
    # Generate states in Orekit format
    states_ = [
        TimeStampedPVCoordinates(
            datetime_to_absolutedate(date),
            Vector3D(*state[0:3].tolist()),
            Vector3D(*state[3:6].tolist()),
        )
        for date, state in zip(dates, states)
    ]

    # Create satellite
    satellite = ObservableSatellite(0)

    # Declare observation list
    observations = []

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

        # Append to observation list
        observations.append(pv)

    # Return observations
    return observations
