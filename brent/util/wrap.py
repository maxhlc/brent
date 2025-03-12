# Standard imports
from typing import overload

# Third-party imports
import numpy as np


class Wrap:

    @overload
    @staticmethod
    def full(x: float, period: float = 2 * np.pi) -> float: ...

    @overload
    @staticmethod
    def full(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray: ...

    @staticmethod
    def full(x: float | np.ndarray, period: float = 2 * np.pi) -> float | np.ndarray:
        # Return wrapped values
        return x % period

    @overload
    @staticmethod
    def half(x: float, period: float = 2 * np.pi) -> float: ...

    @overload
    @staticmethod
    def half(x: np.ndarray, period: float = 2 * np.pi) -> np.ndarray: ...

    @staticmethod
    def half(x: float | np.ndarray, period: float = 2 * np.pi) -> float | np.ndarray:
        # Wrap values by period
        xw = Wrap.full(x, period)

        if isinstance(xw, float):
            # Shift if above half period
            if xw > period / 2:
                xw -= period
        elif isinstance(xw, np.ndarray):
            # Shift values above half period
            xw[xw > period / 2] -= period
        else:
            raise RuntimeError("Unknown type")

        # Return wrapped values
        return xw
