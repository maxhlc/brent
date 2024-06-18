# Standard imports
from typing import Dict, Type

# Internal imports
from .models import BiasModel, SimplifiedAlongtrackSinusoidal

# Declare bias mapping
BIAS_TYPES: Dict[str, Type[BiasModel]] = {
    "none": BiasModel,
    "simplifiedalongtracksinusoidal": SimplifiedAlongtrackSinusoidal,
}


def get_bias_type(name: str) -> Type[BiasModel]:
    # Extract bias type
    biasType = BIAS_TYPES.get(name, None)

    # Raise error for unknown bias type
    if biasType is None:
        raise ValueError(f"Unknown bias type: {biasType}")

    # Return bias type
    return biasType
