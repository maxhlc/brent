from abc import ABC, abstractmethod

# Third-party imports
import numpy as np


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
