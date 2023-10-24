# Standard imports
import os.path

# Define directory paths
BRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BRENT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_OREKIT_DIR = os.path.join(DATA_DIR, "orekit")
