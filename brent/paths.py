# Standard imports
import os.path

# Define directory paths
BRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BRENT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
# Orekit paths
DATA_OREKIT_DIR = os.path.join(DATA_DIR, "orekit")
# THALASSA paths
THALASSA_DIR = "/home/mih21/phd/thalassa-github-maxhlc/"  # TODO: update
THALASSA_LIB_DIR = os.path.join(THALASSA_DIR, "lib")
THALASSA_DATA_DIR = os.path.join(THALASSA_DIR, "data")
THALASSA_DATA_PHYSICAL = os.path.join(THALASSA_DATA_DIR, "physical_constants.txt")
THALASSA_DATA_EARTH = os.path.join(THALASSA_DATA_DIR, "earth_potential", "GRIM5-S1.txt")
THALASSA_DATA_KERNEL = os.path.join(THALASSA_DATA_DIR, "kernels_to_load.furnsh")
