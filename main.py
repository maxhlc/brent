# Standard imports
import argparse

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

# Set figure size
FIGSIZE = (5.5, 4.0)


def main(args):
    # Internal imports
    import brent

    # Set dates
    fitStartDate = args.start
    fitDuration = args.duration
    fitEndDate = fitStartDate + fitDuration

    # Set sample dates
    dates = pd.date_range(fitStartDate, fitEndDate, periods=100)

    # Load TLE propagator
    tlePropagator = brent.io.load_tle_propagator(args.tle, dates[0], dates[-1])

    # Generate pseudo-observation states
    sampleStates = brent.propagators.Propagator(tlePropagator).propagate(dates)

    # Create filter
    filter = brent.filter.OrekitBatchLeastSquares(
        dates,
        sampleStates,
        brent.filter.RTNCovarianceProvider(
            np.array([0.43e3, 5.7e3, 0.17e3, 7.0, 0.43, 0.19])
        ),
    )

    # Execute filter
    fitPropagator = filter.estimate()

    # Get estimated covariance
    fitCovariance = filter.covariance()

    # Generate fit states
    fitStates = fitPropagator.propagate(dates)

    # Calculate RTN transformations
    RTN = brent.frames.rtn(fitStates)

    # Transform fit covariance to RTN
    fitCovarianceRTN = RTN[0, :, :] @ fitCovariance @ RTN[0, :, :].T

    # Calculate state residuals
    deltaStates = sampleStates - fitStates

    # Transform state residuals to RTN
    deltaStatesRTN = np.einsum("ijk,ik -> ij", RTN, deltaStates)

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
    plt.savefig(f"{args.output}_{args.start}_{args.duration}_xyz_pos.png")

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
    plt.savefig(f"{args.output}_{args.start}_{args.duration}_xyz_vel.png")

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
    plt.savefig(f"{args.output}_{args.start}_{args.duration}_rtn_pos.png")

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
    plt.savefig(f"{args.output}_{args.start}_{args.duration}_rtn_vel.png")


if __name__ == "__main__":
    # Parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=lambda x: datetime.strptime(x, "%Y-%m-%d"), required=True)
    parser.add_argument("--duration", type=lambda x: timedelta(float(x)), required=True)
    parser.add_argument("--tle", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Execute
    main(args)
