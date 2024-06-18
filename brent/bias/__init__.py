# Standard imports
from typing import Any, Dict, Type

# Internal imports
from .models import BiasModel, SimplifiedAlongtrackSinusoidal

# Declare bias mapping
BIAS_TYPES: Dict[str, Type[BiasModel]] = {
    type.type: type for type in [BiasModel, SimplifiedAlongtrackSinusoidal]
}


def get_bias_type(name: str) -> Type[BiasModel]:
    # Extract bias type
    biasType = BIAS_TYPES.get(name, None)

    # Raise error for unknown bias type
    if biasType is None:
        raise ValueError(f"Unknown bias type: {name}")

    # Return bias type
    return biasType


def deserialise_bias(struct: Dict[str, Any]) -> BiasModel:
    # Extract bias type
    biasType = get_bias_type(struct["type"])

    # Return deserialised bias
    return biasType.deserialise(struct)
