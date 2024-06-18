# Standard imports
from typing import Dict, Type

# Internal imports
from .models import CovarianceProvider, RTNCovarianceProvider

# Declare noise mapping
NOISE_TYPES: Dict[str, Type[CovarianceProvider]] = {
    "none": CovarianceProvider,
    "rtn": RTNCovarianceProvider,
}


def get_noise_type(name: str) -> Type[CovarianceProvider]:
    # Extract noise type
    noiseType = NOISE_TYPES.get(name, None)

    # Raise error for unknown noise type
    if noiseType is None:
        raise ValueError(f"Unknown noise type: {noiseType}")

    # Return noise type
    return noiseType
