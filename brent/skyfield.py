# Third-party imports
from skyfield.api import Loader
from skyfield.jpllib import SpiceKernel

# Internal imports
from .paths import SKYFIELD_DIR


class Skyfield:
    # Kernel loader
    LOADER = Loader(SKYFIELD_DIR)

    # Kernel objects
    OBJECTS: SpiceKernel = LOADER("de421.bsp")
