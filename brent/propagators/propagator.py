# Future imports
from __future__ import annotations

# Standard imports
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate

# Internal imports
from brent import Constants
from brent.bias import BiasModel
from brent.noise import CovarianceProvider


class Propagator(ABC):
    # Metadata
    type: str

    def __init__(self, bias: BiasModel, noise: CovarianceProvider) -> None:
        # Store bias and noise models
        self.bias = bias
        self.noise = noise

    @abstractmethod
    def _propagate(self, date, frame=Constants.DEFAULT_ECI) -> np.ndarray: ...

    def propagate(self, dates, frame=Constants.DEFAULT_ECI) -> np.ndarray:
        # Propagate biased states
        biased = np.array([self._propagate(date, frame) for date in dates])

        # Calculate unbiased state
        unbiased = self.bias.debias(dates, biased)

        # Return unbiased states array
        return unbiased

    @abstractmethod
    def serialise_parameters(self) -> Dict[str, Any]: ...

    def serialise(self) -> Dict[str, Any]:
        # Return serialised object
        return {
            "type": self.type,
            "parameters": self.serialise_parameters(),
            "bias": self.bias.serialise(),
            "noise": self.noise.serialise(),
        }

    @staticmethod
    @abstractmethod
    def deserialise(struct: Dict[str, Any]) -> Propagator: ...

    # TODO: find way to better centralise deserialisation


class WrappedPropagator(Propagator):
    # Set metadata
    type: str = "wrapped"

    def __init__(
        self,
        propagator,
        bias: BiasModel = BiasModel(),
        noise: CovarianceProvider = CovarianceProvider(),
    ):
        # Store propagator
        self.propagator = propagator

        # Initialise parent
        super().__init__(bias, noise)

    def _propagate(self, date, frame=Constants.DEFAULT_ECI) -> np.ndarray:
        # Convert date to Orekit format
        date_ = datetime_to_absolutedate(date)

        # Propagate state
        state_ = self.propagator.getPVCoordinates(date_, frame)

        # Extract position and velocity
        pos = state_.getPosition().toArray()
        vel = state_.getVelocity().toArray()

        # Concatenate position and velocity
        state = np.array(pos + vel)

        # Return state vector
        return state

    def serialise_parameters(self) -> Tuple[str, Dict[str, Any]]:
        raise ValueError("Unable to serialise WrappedPropagator")

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> WrappedPropagator:
        raise ValueError("Unable to deserialise WrappedPropagator")
