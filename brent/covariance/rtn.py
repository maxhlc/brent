# Third-party imports
import numpy as np

# Internal imports
from .covariance import Covariance
from .factory import CovarianceFactory
import brent.frames


@CovarianceFactory.register("rtn")
class RTNCovariance(Covariance):
    def __init__(self, sigma: list | np.ndarray) -> None:
        # Store standard deviations
        self.sigma = np.array(sigma)

        # Create RTN covariance matrix
        self.covarianceRTN = np.diag(sigma) ** 2

    def _covariance(self, state: np.ndarray) -> np.ndarray:
        # Calculate RTN transform
        # TODO: vectorise for multiple states?
        rtn = brent.frames.RTN.getTransform(state.reshape((1, 6)))[0, :, :]

        # Rotate covariance matrix to inertial frame
        covarianceXYZ = rtn.T @ self.covarianceRTN @ rtn

        # Return inertial covariance
        return covarianceXYZ
