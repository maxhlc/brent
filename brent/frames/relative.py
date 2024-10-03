# Third-party imports
import numpy as np


class RTN:

    @staticmethod
    def getTransform(states: np.ndarray) -> np.ndarray:
        # Extract reference position and velocity vectors
        rRef = states[:, 0:3]
        vRef = states[:, 3:6]

        # Calculate reference angular momentum vectors
        hRef = np.cross(rRef, vRef)

        # Calculate magnitudes
        rRefMag = np.linalg.norm(rRef, axis=1, keepdims=True)
        hRefMag = np.linalg.norm(hRef, axis=1, keepdims=True)

        # Calculate RTN components
        r = rRef / rRefMag
        n = hRef / hRefMag
        t = np.cross(n, r)

        # Create RTN matrix
        rtn = np.stack((r, t, n), axis=1)

        # Expand matrix for combined position and velocity rotation
        rtn = np.kron(np.eye(2, dtype=int), rtn)

        # Return transformation matrix
        return rtn

    @staticmethod
    def transform(
        matrix: np.ndarray,
        vectors: np.ndarray,
        reverse: bool = False,
    ) -> np.ndarray:
        # Reverse transformation (if specified)
        matrix_ = matrix if not reverse else matrix.swapaxes(1, 2)

        # Return transformed vectors
        return np.einsum("ijk,ik -> ij", matrix_, vectors)
