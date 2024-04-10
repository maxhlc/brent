# Orekit imports
import orekit
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions, Constants

# Default parameters
DEFAULT_IERSCONVENTIONS = IERSConventions.IERS_2010
DEFAULT_ECI = FramesFactory.getEME2000()
DEFAULT_ECEF = FramesFactory.getITRF(DEFAULT_IERSCONVENTIONS, True)
DEFAULT_MU = Constants.EIGEN5C_EARTH_MU
DEFAULT_RADIUS = Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS
