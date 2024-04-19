# Orekit imports
import orekit
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions, Constants as OrekitConstants


class Constants:
    # Default conventions
    DEFAULT_IERSCONVENTIONS = IERSConventions.IERS_2010

    # Default Earth frames
    DEFAULT_ECI = FramesFactory.getEME2000()
    DEFAULT_ECEF = FramesFactory.getITRF(DEFAULT_IERSCONVENTIONS, True)

    # Default Earth properties
    DEFAULT_MU = OrekitConstants.EIGEN5C_EARTH_MU
    DEFAULT_RADIUS = OrekitConstants.IERS2010_EARTH_EQUATORIAL_RADIUS
    DEFAULT_FLATTENING = OrekitConstants.IERS2010_EARTH_FLATTENING
