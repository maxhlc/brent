# Orekit imports
import orekit
from org.orekit.propagation.analytical import AggregateBoundedPropagator
from org.orekit.propagation.analytical.tle import TLEPropagator
import java.util


def tles_to_propagator(tles):
    # Check for number of TLEs
    if len(tles) < 1:
        raise ValueError("Insufficent number of TLEs")

    # TODO: sort TLEs to ensure in order?

    # Declare map for propagators
    propagatorMap = java.util.TreeMap()

    # Iterate through TLEs
    for tle in tles:
        # Extract epoch date
        epoch = tle.getDate()

        # Create propagator
        propagator = TLEPropagator.selectExtrapolator(tle)

        # Add to map
        propagatorMap.put(epoch, propagator)

    # Extract start and end dates
    # TODO: review dates
    dateStart = tles[0].getDate()
    dateEnd = tles[-1].getDate()

    # Return aggregate propagator
    return AggregateBoundedPropagator(propagatorMap, dateStart, dateEnd)
