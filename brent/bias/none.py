# Third-party imports
import numpy as np

# Internal imports
from .factory import BiasFactory
from .bias import Bias


@BiasFactory.register("none")
class NoneBias(Bias):

    def biases(self, dates, states) -> np.ndarray:
        # Return zero biases
        return np.zeros(states.shape)

    @classmethod
    def fit(cls, dates, states, reference, p0):
        return NoneBias()
