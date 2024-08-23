# Third-party imports
import numpy as np
import scipy.linalg

# Internal imports
import brent.frames


class CovarianceProvider:
    def __init__(self):
        pass

    def _covariance(self, state: np.ndarray) -> np.ndarray:
        # Return identity matrix
        return np.identity(len(state))

    def covariance(self, states: np.ndarray) -> np.ndarray:
        if len(states.shape) == 1:  # Single state
            # Return single covariance matrix
            return self._covariance(states)
        else:  # Multiple states
            # Return block diagonal covariance matrix
            return scipy.linalg.block_diag(
                *[self._covariance(state) for state in states]
            )


class RTNCovarianceProvider(CovarianceProvider):
    def __init__(self, sigma: np.ndarray) -> None:
        # Store standard deviations
        self.sigma = sigma

        # Create RTN covariance matrix
        self.covarianceRTN = np.diag(sigma) ** 2

    def _covariance(self, state: np.ndarray) -> np.ndarray:
        # Calculate RTN transform
        # TODO: vectorise for multiple states?
        rtn = brent.frames.rtn(state.reshape((1, 6)))[0, :, :]

        # Rotate covariance matrix to inertial frame
        covarianceXYZ = rtn.T @ self.covarianceRTN @ rtn

        # Return inertial covariance
        return covarianceXYZ
