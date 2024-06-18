# Future imports
from __future__ import annotations

# Standard imports
from typing import Any, Dict

# Third-party imports
import numpy as np

# Internal imports
import brent.frames


class CovarianceProvider:
    # Set metadata
    type: str = "None"

    def __init__(self):
        pass

    def __call__(self, state):
        # Return identity matrix
        return np.identity(len(state))

    def serialise(self):
        # Return serialised model
        return {
            "type": self.type,
            "parameters": {},
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> CovarianceProvider:
        # Assert type and parameters match
        assert struct["type"] == CovarianceProvider.type
        assert struct["parameters"] == {}

        # Return covariance model
        return CovarianceProvider()


class RTNCovarianceProvider(CovarianceProvider):
    # Set metadata
    type: str = "RTN"

    def __init__(self, sigma: np.ndarray):
        # Assert shape
        assert sigma.shape == (6,)

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

    def serialise(self):
        # Return serialised model
        return {
            "type": self.type,
            "parameters": {
                "std": self.sigma.tolist(),
            },
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> RTNCovarianceProvider:
        # Assert type matches
        assert struct["type"] == RTNCovarianceProvider.type

        # Extract standard deviations
        sigma = np.array(struct["parameters"]["std"])

        # Return covariance model
        return RTNCovarianceProvider(sigma)
