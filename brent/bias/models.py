# Future imports
from __future__ import annotations

# Standard imports
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

# Third-party imports
import numpy as np

# Internal imports
import brent.frames


class BiasModel:
    # Set metadata
    type: str = "None"

    def biases(self, dates, states):
        # Return zero biases
        return np.zeros(states.shape)

    def debias(self, dates, states):
        # Calculate biases
        bias = self.biases(dates, states)

        # Debias states
        states_debiased = states - bias

        # Return debiased states
        return states_debiased

    def serialise(self):
        # Return serialised model
        return {
            "type": self.type,
            "parameters": {},
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> BiasModel:
        # Assert type and parameters match
        assert struct["type"] == BiasModel.type
        assert struct["parameters"] == {}

        # Return bias model
        return BiasModel()


@dataclass
class SimplifiedAlongtrackSinusoidal(BiasModel):
    # Model parameters
    amplitude: float
    frequency: float
    phase: float
    offset: float

    # Set metadata
    type: str = "SimplifiedAlongtrackSinusoidal"

    def __model(self, t: np.ndarray):
        # Calculate model period
        period = 2.0 * np.pi / self.frequency

        # Return along-track bias
        return self.amplitude * np.sin(period * (t + self.phase)) + self.offset

    def biases(self, dates, states):
        # Calculate radial distances
        rmag = np.linalg.norm(states[:, 0:3], axis=1)

        # Find RTN transform
        RTN = brent.frames.rtn(states)
        RTNt = RTN.swapaxes(1, 2)

        # Calculate offset dates
        dfunc = np.vectorize(lambda x: x / np.timedelta64(1, "D"))
        t = dfunc(dates - datetime(2022, 1, 1))

        # Calculate bias
        bias_RTN = np.zeros(states.shape)
        bias_RTN[:, 1] = self.__model(t) * rmag

        # Rotate to inertial frame
        bias = np.einsum("ijk,ik -> ij", RTNt, bias_RTN)

        # Return biases
        return bias

    def serialise(self):
        # Return serialised model
        return {
            "type": self.type,
            "parameters": {
                "amplitude": self.amplitude,
                "frequency": self.frequency,
                "phase": self.phase,
                "offset": self.offset,
            },
        }

    @staticmethod
    def deserialise(struct) -> SimplifiedAlongtrackSinusoidal:
        # Assert type matches
        assert struct["type"] == SimplifiedAlongtrackSinusoidal.type

        # Extract model parameters
        amplitude = struct["parameters"]["amplitude"]
        frequency = struct["parameters"]["frequency"]
        phase = struct["parameters"]["phase"]
        offset = struct["parameters"]["offset"]

        # Return bias model
        return SimplifiedAlongtrackSinusoidal(amplitude, frequency, phase, offset)
