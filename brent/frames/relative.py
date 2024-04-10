# Third-party imports
import numpy as np


def rtn(states):
    # Extract reference position and velocity vectors
    rRef = states[:, 0:3]
    vRef = states[:, 3:6]

    # Calculate reference angular momentum vectors
    hRef = np.cross(rRef, vRef)

    # Calculate magnitudes
    rRefMag = np.linalg.norm(rRef, axis=1, keepdims=True)
    hRefMag = np.linalg.norm(hRef, axis=1, keepdims=True)

    # Calculate RTN components
    R = rRef / rRefMag
    N = hRef / hRefMag
    T = np.cross(N, R)

    # Create RTN matrix
    RTN = np.stack((R, T, N), axis=1)

    # Expand matrix for combined position and velocity rotation
    RTN = np.kron(np.eye(2, dtype=int), RTN)

    # Return transformation matrix
    return RTN
