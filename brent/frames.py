# Third-party imports
import numpy as np


def rtn(statesRef, states):
    # Extract reference position and velocity vectors
    rRef = statesRef[:, 0:3]
    vRef = statesRef[:, 3:6]

    # Calculate reference angular momentum vectors
    hRef = np.cross(rRef, vRef)

    # Calculate magnitudes
    rRefMag = np.linalg.norm(rRef, axis=1).reshape((-1, 1))
    hRefMag = np.linalg.norm(hRef, axis=1).reshape((-1, 1))

    # Calculate RTN components
    R = rRef / rRefMag
    N = hRef / hRefMag
    T = np.cross(N, R)

    # Create RTN matrix
    RTN = np.stack((R, T, N), axis=1)

    # Expand matrix for combined position and velocity rotation
    RTN = np.kron(np.eye(2, dtype=int), RTN)

    # Calculate state differences
    stateDeltas = states - statesRef

    # Rotate state differences to RTN frame
    stateDeltasRTN = np.squeeze(RTN @ stateDeltas[:, :, None])

    # Return RTN differences
    return stateDeltasRTN
