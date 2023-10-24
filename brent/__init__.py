# Internal imports
from . import io
from . import paths

# Orekit imports
import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Initialise Orekit
vm = orekit.initVM()
setup_orekit_curdir(paths.DATA_OREKIT_DIR)
