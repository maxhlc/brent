# Third-party imports
import numpy as np


class Weight:
    def __init__(self):
        pass

    def __call__(self, y):
        # Return identity matrix
        return np.identity(len(y))


class ConstantWeight(Weight):
    def __init__(self, nvar, weights):
        # Check for consistency
        assert nvar == len(weights.ravel())

        # Store number of variables and weights
        self.nvar = nvar
        self.weights = weights.ravel()

    def __call__(self, y):
        # Calculate number of residuals
        ny = len(y)
        nvar = self.nvar
        nsam = ny // nvar

        # Declare weight matrix
        W = np.identity(len(y))

        # Update weight matrix
        W[np.diag_indices_from(W)] = np.tile(self.weights, nsam)

        # Return weight matrix
        return W


class SampledWeight(Weight):
    def __init__(self, nvar):
        # Store number of variables
        self.nvar = nvar

    def __call__(self, y):
        # Calculate number of residuals
        ny = len(y)
        nvar = self.nvar
        nsam = ny // nvar

        # Declare weight matrix
        W = np.identity(len(y))

        # Calculate residual covariance matrix and extract diagonal terms
        yCov = np.cov(y.reshape((-1, self.nvar)), rowvar=False)
        ySig = np.sqrt(np.diag(yCov))

        # Update weight matrix
        W[np.diag_indices_from(W)] /= np.tile(ySig, nsam)

        # Return weight matrix
        return W


class CovarianceProviderWeight(Weight):
    def __init__(self, states, covarianceProvider):
        # Store number of variables
        self.states = states

        # Extract number of states and variables
        self.nstates = states.shape[0]
        self.nvar = states.shape[1]

        # Calculate number of elements in residual vector
        self.nelem = self.nstates * self.nvar

        # Calculate variances
        self.variances = np.concatenate(
            [np.diag(covarianceProvider(state)) for state in states]
        )

        # Calculate weight matrix
        self.W = np.diag(1.0 / np.sqrt(self.variances))

    def __call__(self, y):
        # Check number of residuals
        assert len(y) == self.nelem

        # Return weight matrix
        return self.W
