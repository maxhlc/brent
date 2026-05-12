# Third-party imports
import numpy as np

# Internal imports
from .covariance import Covariance
from .factory import CovarianceFactory


@CovarianceFactory.register("none")
class IdentityCovariance(Covariance):

    def _covariance(self, state: np.ndarray) -> np.ndarray:
        # Return identity matrix
        return np.identity(len(state))
