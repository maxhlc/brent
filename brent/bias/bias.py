# Future imports
from __future__ import annotations

# Standard imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np
import scipy.optimize


# TODO: move to more generic location
def rmse(vectors):
    return np.sqrt(np.mean(np.linalg.norm(vectors, axis=1) ** 2))


class Bias(ABC):

    @abstractmethod
    def biases(self, dates, states) -> np.ndarray: ...

    def debias(self, dates, states) -> np.ndarray:
        # Calculate biases
        bias = self.biases(dates, states)

        # Debias states
        states_debiased = states - bias

        # Return debiased states
        return states_debiased

    @classmethod
    def fit(cls, dates, states, reference, p0) -> Bias:
        # Fit wrapper function
        def func(_, *p):
            # Create bias model
            model = cls(*p)

            # Debias states
            states_debiased = model.debias(dates, states)

            # Return debiased states
            return states_debiased.ravel()

        # Extract fit data
        y = reference.ravel()
        x = np.zeros(y.shape)

        # Fit model
        popt, _ = scipy.optimize.curve_fit(func, x, y, p0)

        # Returned fitted bias model
        return cls(*popt)

    def evaluate(self, dates, states, reference):
        # Calculate debiased states
        states_debiased = self.debias(dates, states)

        # Calculate state deltas
        states_delta = states_debiased - reference

        # Calculate RMSEs
        position_rmse = rmse(states_delta[:, 0:3])
        velocity_rmse = rmse(states_delta[:, 3:6])

        # Return RMSEs
        return position_rmse, velocity_rmse
