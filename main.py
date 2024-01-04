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
    sampleStates = tlePropagator.propagate(dates)

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

    # Calculate residual sample covariances
    residualCovariance = np.cov(deltaStates, rowvar=False)
    residualCovarianceRTN = np.cov(deltaStatesRTN, rowvar=False)

    # Load SP3 data
    testPropagator = brent.io.load_sp3_propagator(args.sp3, args.sp3name)

    # Create test dates
    testDates = pd.date_range(fitEndDate, fitEndDate + timedelta(30), freq="0.25D")

    # Calculate test states
    sampleTestStates = tlePropagator.propagate(testDates)
    fitTestStates = fitPropagator.propagate(testDates)
    testStates = testPropagator.propagate(testDates)

    # Calculate position error
    sampleError = np.linalg.norm(sampleTestStates - testStates, axis=1)
    fitError = np.linalg.norm(fitTestStates - testStates, axis=1)

    # Calculate proportion of fit period where the fit outperforms the samples
    proportion = np.mean(fitError <= sampleError)


    # Plot results
    if args.plot:
        # Generate output path prefix
        plotPathPrefix = f"{args.output}_{args.start.strftime('%Y-%m-%d')}_{args.duration.days}"

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
        plt.savefig(f"{plotPathPrefix}_xyz_pos.png")
        plt.close()

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
        plt.savefig(f"{plotPathPrefix}_xyz_vel.png")
        plt.close()

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
        plt.savefig(f"{plotPathPrefix}_rtn_pos.png")
        plt.close()

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
        plt.savefig(f"{plotPathPrefix}_rtn_vel.png")
        plt.close()

    # Print results
    if args.verbose:
        # Update linewidth
        np.set_printoptions(linewidth=160)

        # Calculate fit statistics
        fitXYZsigma = np.sqrt(np.diag(fitCovariance))
        residualXYZsigma = np.sqrt(np.diag(np.cov(deltaStates, rowvar=False)))
        fitRTNsigma = np.sqrt(np.diag(fitCovarianceRTN))
        residualRTNsigma = np.sqrt(np.diag(np.cov(deltaStatesRTN, rowvar=False)))

        # Print statistics
        print(f"Fit XYZ 1-sigma:      {fitXYZsigma}")
        print(f"Residual XYZ 1-sigma: {residualXYZsigma}")
        print(f"XYZ 1-sigma ratio:    {fitXYZsigma / residualXYZsigma}")
        print("")
        print(f"Fit RTN 1-sigma:      {fitRTNsigma}")
        print(f"Residual RTN 1-sigma: {residualRTNsigma}")
        print(f"RTN 1-sigma ratio:    {fitRTNsigma / residualRTNsigma}")

    # Return results
    return {
        "dates": dates,
        "sampleStates": sampleStates,
        "fitStates": fitStates,
        "deltaStates": deltaStates,
        "deltaStatesRTN": deltaStatesRTN,
        "fitCovariance": fitCovariance,
        "fitCovarianceRTN": fitCovarianceRTN,
        "residualCovariance": residualCovariance,
        "residualCovarianceRTN": residualCovarianceRTN,
        "testDates": testDates,
        "sampleTestStates": sampleTestStates,
        "fitTestStates": fitTestStates,
        "testStates": testStates,
        "sampleError": sampleError,
        "fitError": fitError,
        "proportion": proportion,
    }


if __name__ == "__main__":
    # Parse inputs
    # TODO: fix format
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=lambda x: datetime.strptime(x, "%Y-%m-%d"), required=True)
    parser.add_argument("--duration", type=lambda x: timedelta(float(x)), required=True)
    parser.add_argument("--tle", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sp3", required=True)
    parser.add_argument("--sp3name", required=True)
    args = parser.parse_args()

    # Execute
    main(args)
