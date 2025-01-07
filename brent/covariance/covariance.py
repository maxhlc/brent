# Standard imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import scipy.linalg


class Covariance(ABC):

    @abstractmethod
    def _covariance(self, state: np.ndarray) -> np.ndarray: ...

    def covariance(self, states: np.ndarray) -> np.ndarray:
        if len(states.shape) == 1:  # Single state
            # Return single covariance matrix
            return self._covariance(states)
        else:  # Multiple states
            # Return block diagonal covariance matrix
            return scipy.linalg.block_diag(
                *[self._covariance(state) for state in states]
            )
