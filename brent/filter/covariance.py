# Third-party imports
import numpy as np

# Internal imports
import brent.frames


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
