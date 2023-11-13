# Load paths
from . import paths

# Orekit imports
import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Initialise Orekit
vm = orekit.initVM()
setup_orekit_curdir(paths.DATA_OREKIT_DIR)

# Internal imports
from . import filter
from . import frames
from . import io
from . import propagators
from . import util
