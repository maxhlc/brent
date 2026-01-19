# TODO: docstrings
# TODO: type hinting
# TODO: standardise datetime/timedelta type (probably NumPy)
# TODO: fix bug where THALASSA breaks following the use of Matplotlib

# Standard imports
import sys

# Load paths
from . import paths

# Orekit imports
import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Initialise Orekit
vm = orekit.initVM()
setup_orekit_curdir(paths.DATA_OREKIT_DIR)

# Add PyTHALASSA to path
sys.path.append(paths.THALASSA_LIB_DIR)

# Load constants
from .constants import Constants

# Internal imports
from . import bias
from . import covariance
from . import filter
from . import frames
from . import io
from . import propagators
from . import util
from . import skyfield
