# Third-party imports
import numpy as np
import scipy.linalg

# Internal imports
import brent.frames


class CovarianceProvider:
    def __init__(self):
        pass

    def __call__(self, state: np.ndarray) -> np.ndarray:
        # Return identity matrix
        return np.identity(len(state))

    def stddev(self, state: np.ndarray) -> np.ndarray:
        # Calculate standard deviation matrix
        # NOTE: non-diagonal elements are discarded
        sigma = np.sqrt(np.diag(np.diag(self(state))))

        # Return standard deviation matrix
        return sigma

    def stddevs(self, states: np.ndarray) -> np.ndarray:
        # Generate standard deviation matrices for each state
        sigmas = [self.stddev(state) for state in states]

        # Create block diagonal matrix
        sigma = scipy.linalg.block_diag(*sigmas)

        # Return standard deviation matrix
        return sigma


class RTNCovarianceProvider(CovarianceProvider):
    def __init__(self, sigma: np.ndarray) -> None:
        # Store standard deviations
        self.sigma = sigma

        # Create RTN covariance matrix
        self.covarianceRTN = np.diag(sigma) ** 2

    def __call__(self, state: np.ndarray) -> np.ndarray:
        # Calculate RTN transform
        rtn = brent.frames.rtn(state.reshape((1, 6)))[0, :, :]

        # Rotate covariance matrix to inertial frame
        covarianceXYZ = rtn.T @ self.covarianceRTN @ rtn

        # Return inertial covariance
        return covarianceXYZ
