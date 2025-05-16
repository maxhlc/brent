# Standard imports
from enum import Enum

# Orekit imports
import orekit
from org.orekit.orbits import PositionAngleType


class AngleType(Enum):
    TRUE = PositionAngleType.TRUE
    MEAN = PositionAngleType.MEAN
    ECCENTRIC = PositionAngleType.ECCENTRIC
