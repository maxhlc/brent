# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

# Internal imports
import brent

# Set figure size
FIGSIZE = (5.5, 4.0)


def main():
    # Set dates
    fitStartDate = datetime(2022, 6, 1)
    fitDuration = timedelta(7)
    fitEndDate = fitStartDate + fitDuration

    # Set sample dates
    dates = pd.date_range(fitStartDate, fitEndDate, periods=100)

    # Load TLE propagator
    tlePropagator = brent.io.load_tle_propagator("./data/tle/8820.json", dates[0], dates[-1])

    # Generate pseudo-observation states
    sampleStates = brent.propagators.Propagator(tlePropagator).propagate(dates)

    # Calculate RTN transform
    samplesRTN = brent.frames.rtn(sampleStates)

    # Create initial Orekit propagator
    fitPropagator_ = brent.propagators.default_propagator(dates[0], sampleStates[0, :])

    # Wrap Orekit propagator
    fitPropagator = brent.propagators.Propagator(fitPropagator_)

    # Declare error function
    def func(x):
        # Update initial state
        fitPropagator.setInitialState(dates[0], x)

        # Propagate state
        states = fitPropagator.propagate(dates)

        # Calculate position errors
        error = states - sampleStates

        # Return 1D array of errors
        return error.ravel()

    # Declare weight function
    wfunc = brent.filter.SampledWeight(6)

    # Create filter
    filter = brent.filter.BatchLeastSquares(func, wfunc)

    # Execute filter
    fitState = filter.estimate(sampleStates[0, :])
    fitPropagator.setInitialState(dates[0], fitState)

    # Generate fit states
    fitStates = fitPropagator.propagate(dates)

    # Calculate state residuals
    deltaStates = fitStates - sampleStates

    # Transform state residuals to RTN
    deltaStatesRTN = np.einsum("ijk,ik -> ij", samplesRTN, deltaStates)

    # Plot inertial position residuals
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.plot(dates, deltaStates[:, 0:3], label=["X", "Y", "Z"])
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
    plt.plot(dates, deltaStates[:, 3:6], label=["X", "Y", "Z"])
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
    plt.plot(dates, deltaStatesRTN[:, 0:3], label=["R", "T", "N"])
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
    plt.plot(dates, deltaStatesRTN[:, 3:6], label=["R", "T", "N"])
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
