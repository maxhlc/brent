# Standard imports
from abc import ABC, abstractmethod

# Third-party imports
import numpy as np

# Internal imports
from brent.propagators import Propagator, NumericalPropagatorParameters


class BatchLeastSquares(ABC):

    @abstractmethod
    def estimate(self) -> Propagator: ...

    @abstractmethod
    def getEstimatedState(self) -> np.ndarray: ...

    @abstractmethod
    def getEstimatedCovariance(self) -> np.ndarray: ...

    @abstractmethod
    def getEstimatedModel(self) -> NumericalPropagatorParameters: ...
