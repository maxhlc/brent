# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate as absolutedate

# Internal imports
import brent

# Set figure size
FIGSIZE = (5.5, 4.0)


def main():
    # Load TLEs
    tles = brent.io.load_tle("./data/tle/8820.json")

    # Set dates
    fitStartDate = datetime(2022, 6, 1)
    fitDuration = timedelta(7)
    fitEndDate = fitStartDate + fitDuration

    # Set sample dates
    sampleDates_ = pd.date_range(fitStartDate, fitEndDate, periods=100)
    sampleDates = [absolutedate(date) for date in sampleDates_]

    # Extract TLE subset
    tles_ = [
        tle
        for tle in tles
        if (tle.getDate().durationFrom(absolutedate(fitStartDate)) >= 0)
        and (tle.getDate().durationFrom(absolutedate(fitEndDate)) <= 0)
    ]

    # Create TLE propagator
    tlePropagator = brent.propagators.tles_to_propagator(tles_)

    # Generate pseudo-observation states
    sampleStates = brent.propagators.propagate(tlePropagator, sampleDates)

    # Create filter
    filter = brent.filter.BatchLeastSquares(sampleStates)

    # Execute filter
    fitPropagator = filter.estimate()

    # Generate fit states
    fitStates = brent.propagators.propagate(fitPropagator, sampleDates)

    # Convert states to NumPy arrays
    _, sampleStates_ = brent.propagators.pv_to_array(sampleStates)
    fitDates, fitStates_ = brent.propagators.pv_to_array(fitStates)

    # Calculate state residuals
    deltaStates = fitStates_ - sampleStates_

    # Transform state residuals to RTN
    RTN = brent.frames.rtn(sampleStates_)
    deltaStatesRTN = np.einsum("ijk,ik -> ij", RTN, deltaStates)

    # Plot inertial position residuals
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.plot(fitDates, deltaStates[:, 0:3], label=["X", "Y", "Z"])
    plt.xlabel("Date [UTC]")
    plt.ylabel("Residual Error [m]")
    plt.legend()
    fig.autofmt_xdate()
    plt.grid(which="both", alpha=0.5)
    plt.grid(which="minor", alpha=0.25)
    plt.tight_layout()
    plt.savefig("./output/xyz.png")

    # Plot inertial velocity residuals
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.plot(fitDates, deltaStates[:, 3:6], label=["X", "Y", "Z"])
    plt.xlabel("Date [UTC]")
    plt.ylabel("Residual Error [m/s]")
    plt.legend()
    fig.autofmt_xdate()
    plt.grid(which="both", alpha=0.5)
    plt.grid(which="minor", alpha=0.25)
    plt.tight_layout()
    plt.savefig("./output/vxyz.png")

    # Plot RTN position residuals
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.plot(fitDates, deltaStatesRTN[:, 0:3], label=["R", "T", "N"])
    plt.xlabel("Date [UTC]")
    plt.ylabel("Residual Error [m]")
    plt.legend()
    fig.autofmt_xdate()
    plt.grid(which="both", alpha=0.5)
    plt.grid(which="minor", alpha=0.25)
    plt.tight_layout()
    plt.savefig("./output/rtn.png")

    # Plot RTN velocity residuals
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.plot(fitDates, deltaStatesRTN[:, 3:6], label=["R", "T", "N"])
    plt.xlabel("Date [UTC]")
    plt.ylabel("Residual Error [m/s]")
    plt.legend()
    fig.autofmt_xdate()
    plt.grid(which="both", alpha=0.5)
    plt.grid(which="minor", alpha=0.25)
    plt.tight_layout()
    plt.savefig("./output/vrtn.png")


if __name__ == "__main__":
    main()
