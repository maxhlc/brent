# Standard imports
from typing import Any, Dict, Type

# Internal imports
from .numerical import NumericalPropagator, NumericalPropagatorParameters
from .propagator import Propagator, WrappedPropagator
from .sp3 import SP3Propagator
from .tle import TLEPropagator

# Declare propagator mapping
PROPAGATOR_TYPES: Dict[str, Type[Propagator]] = {
    "Numerical": NumericalPropagator,
    "TLE": TLEPropagator,
    "SP3": SP3Propagator,
}


def get_propagator_type(name: str) -> Type[Propagator]:
    # Extract propagator type
    propagatorType = PROPAGATOR_TYPES.get(name, None)

    # Raise error for unknown propagator type
    if propagatorType is None:
        raise ValueError(f"Unknown propagator type: {propagatorType}")

    # Return propagator type
    return propagatorType


def deserialise_propagator(struct: Dict[str, Any]) -> Propagator:
    # Extract propagator type
    propagatorType = get_propagator_type(struct["type"])

    # Return deserialised propagator
    return propagatorType.deserialise(propagatorType, struct)
