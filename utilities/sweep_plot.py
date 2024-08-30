# Standard imports
from argparse import ArgumentParser

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set output figure size
FIGSIZE = (5.5, 4.0)

# Declare name map
SP3MAP = {
    None: "Unassigned",  # TODO: replace bodge for objects without SP3 name
    "L51": "LAGEOS-1",
    "L52": "LAGEOS-2",
    "L53": "Etalon 1",
    "L54": "Etalon 2",
}

# Declare function to convert duration to days
duration_to_days = np.vectorize(lambda x: x / np.timedelta64(1, "D"))


def roundup(val: float, interval: float) -> float:
    # Return value, rounded up to specified interval
    return val + (interval - val % interval)


def plot_window_mesh(df: pd.DataFrame, fname: str) -> None:
    # Calculate limits
    # TODO: cleaner implementation
    xlim = (
        np.min(df["midPoint"]),
        np.max(df["midPoint"]),
    )

    # Iterate through objects
    for object in np.unique(df["name"]):
        # Extract corresponding sub-table for object
        df_object = df[df["name"] == object].copy()

        # Extract x-y-z variables
        x = df_object["midPoint"].to_numpy()
        y = duration_to_days(df_object["duration"])
        z = df_object["fitErrorRMS"].to_numpy()

        # Extract reference propagator
        referencePropagators = np.unique(df_object["referencePropagator"])
        referencePropagator = referencePropagators[0]

        # Raise error for multiple reference propagators
        if len(referencePropagators) != 1:
            raise ValueError("Mixture of reference propagators")

        # Find unique x-y coordinates
        xu = np.unique(x)
        yu = np.unique(y)

        # Calculate mesh size
        nx = len(xu)
        ny = len(yu)

        # Iterate through x-y coordinates
        for ix in xu:
            for iy in yu:
                # Extract indicies
                idx = x == ix
                idy = y == iy

                # Find intersection
                idxy = np.logical_and(idx, idy)

                # Calculate number of corresponding points
                n = np.sum(idxy.astype(int))

                # Add NaN point if the point doesn't exist
                if n == 0:
                    x = np.append(x, ix)
                    y = np.append(y, iy)
                    z = np.append(z, np.nan)
                elif n > 1:
                    raise ValueError(f"Repeated point at ({ix}, {iy})")

        # Sort points into mesh
        idx = np.lexsort((y, x)).reshape((nx, ny))

        # Create plot
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Plot mesh
        plt.contourf(x[idx], y[idx], z[idx])

        # Set limits
        # TODO: y- and z-axes
        plt.xlim(xlim)

        # Set axis labels
        plt.xlabel("Fit Window Midpoint [-]")
        plt.ylabel("Fit Window Size [days]")

        # Add colour bar
        plt.colorbar(label=f"Position RMSE (w.r.t. {referencePropagator}) [m]")

        # Format dates
        fig.autofmt_xdate()

        # Add grid
        plt.grid()

        # Set layout
        plt.tight_layout()

        # Export plot
        plt.savefig(f"{fname}_duration_rmse_{object}.png", dpi=600)

        # Close plot
        plt.close()


def plot_sample_mesh(df: pd.DataFrame, fname: str) -> None:
    # Calculate limits
    # TODO: cleaner implementation
    xlim = (
        np.min(df["midPoint"]),
        np.max(df["midPoint"]),
    )

    # Iterate through objects
    for object in np.unique(df["name"]):
        # Extract corresponding sub-table for object
        df_object = df[df["name"] == object].copy()

        # Extract x-y-z variables
        x = df_object["midPoint"].to_numpy()
        y = df_object["samples"].to_numpy()
        z = df_object["fitErrorRMS"].to_numpy()

        # Extract reference propagator
        referencePropagators = np.unique(df_object["referencePropagator"])
        referencePropagator = referencePropagators[0]

        # Raise error for multiple reference propagators
        if len(referencePropagators) != 1:
            raise ValueError("Mixture of reference propagators")

        # Find unique x-y coordinates
        xu = np.unique(x)
        yu = np.unique(y)

        # Calculate mesh size
        nx = len(xu)
        ny = len(yu)

        # Iterate through x-y coordinates
        for ix in xu:
            for iy in yu:
                # Extract indicies
                idx = x == ix
                idy = y == iy

                # Find intersection
                idxy = np.logical_and(idx, idy)

                # Calculate number of corresponding points
                n = np.sum(idxy.astype(int))

                # Add NaN point if the point doesn't exist
                if n == 0:
                    x = np.append(x, ix)
                    y = np.append(y, iy)
                    z = np.append(z, np.nan)
                elif n > 1:
                    raise ValueError(f"Repeated point at ({ix}, {iy})")

        # Sort points into mesh
        idx = np.lexsort((y, x)).reshape((nx, ny))

        # Create plot
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Plot mesh
        plt.contourf(x[idx], y[idx], z[idx])

        # Set limits
        # TODO: y- and z-axes
        plt.xlim(xlim)

        # Set axis labels
        plt.xlabel("Fit Window Midpoint [-]")
        plt.ylabel("Sample Size [-]")

        # Add colour bar
        plt.colorbar(label=f"Position RMSE (w.r.t. {referencePropagator}) [m]")

        # Format dates
        fig.autofmt_xdate()

        # Add grid
        plt.grid()

        # Set layout
        plt.tight_layout()

        # Export plot
        plt.savefig(f"{fname}_samples_rmse_{object}.png", dpi=600)

        # Close plot
        plt.close()


def plot_proportion(df: pd.DataFrame, fname: str) -> None:
    # Iterate through windows
    for window in np.unique(df["duration"]):
        # Iterate through samples
        for samples in np.unique(df["samples"]):
            # Calculate number of days in window
            days = int(window / np.timedelta64(1, "D"))

            # Extract subtable
            idx = np.logical_and(df["duration"] == window, df["samples"] == samples)
            df_ = df[idx].copy()

            # Extract reference propagator
            referencePropagators = np.unique(df_["referencePropagator"])
            referencePropagator = referencePropagators[0]

            # Raise error for multiple reference propagators
            if len(referencePropagators) != 1:
                raise ValueError("Mixture of reference propagators")

            df_ = df_[["testDaysPostFitEnd", "name", "errorDiffBetter"]]

            # Calculate limits
            # TODO: cleaner implementation
            xlim = (
                np.min(df_["testDaysPostFitEnd"].explode(["testDaysPostFitEnd"])),
                np.max(df_["testDaysPostFitEnd"].explode(["testDaysPostFitEnd"])),
            )

            # Expand data frame
            df_ = (
                df_.explode(["testDaysPostFitEnd", "errorDiffBetter"])
                .groupby(["name", "testDaysPostFitEnd"])
                .mean()
            )

            # Create plot
            fig, ax = plt.subplots(figsize=FIGSIZE)

            # Plot position RMSEs
            sns.lineplot(
                data=df_,
                x="testDaysPostFitEnd",
                y="errorDiffBetter",
                hue="name",
            )

            # Set axis labels
            plt.xlabel("Elapsed Time Post-fit [days]")
            plt.ylabel(f"Proportion (w.r.t. {referencePropagator}) [-]")

            # Update legend title
            plt.legend(title="Satellite")

            # Set axis limits
            plt.xlim(xlim)
            plt.ylim((0.0, 1.0))

            # Plot reference lines
            plt.axhline(0.5, color="black")
            plt.axvline(0.0, color="black")

            # Add grid
            plt.grid()

            # Set layout
            plt.tight_layout()

            # Export plot
            plt.savefig(f"{fname}_proportion_{samples}S_{days}D.png", dpi=600)

            # Close plot
            plt.close()


def plot_errors(df: pd.DataFrame, fname: str) -> None:
    # Calculate limits
    # TODO: cleaner implementation, adjust rounding interval based on maximum value
    xlim = (
        np.min(df["midPoint"]),
        np.max(df["midPoint"]),
    )
    ylim = (
        0,
        roundup(
            np.max(df["fitErrorRMS"]),
            5000,
        ),
    )

    # Iterate through windows
    for window in np.unique(df["duration"]):
        # Iterate through samples
        for samples in np.unique(df["samples"]):
            # Calculate number of days in window
            days = int(window / np.timedelta64(1, "D"))

            # Extract subtable
            df_ = df[df["duration"] == window].copy()

            # Sort subtable by object name
            df_ = df_.sort_values("name")

            # Extract reference propagator
            referencePropagators = np.unique(df_["referencePropagator"])
            referencePropagator = referencePropagators[0]

            # Raise error for multiple reference propagators
            if len(referencePropagators) != 1:
                raise ValueError("Mixture of reference propagators")

            # Create plot
            fig, ax = plt.subplots(figsize=FIGSIZE)

            # Plot position RMSEs
            sns.lineplot(data=df_, x="midPoint", y="fitErrorRMS", hue="name")

            # Set axis labels
            plt.xlabel("Fit Window Midpoint [-]")
            plt.ylabel(f"Position RMSE (w.r.t. {referencePropagator}) [m]")

            # Update legend title
            plt.legend(title="Satellite")

            # Set limits
            plt.xlim(xlim)
            plt.ylim(ylim)

            # Format dates
            fig.autofmt_xdate()

            # Add grid
            plt.grid()

            # Set layout
            plt.tight_layout()

            # Export plot
            plt.savefig(f"{fname}_rmse_{samples}S_{days}D.png", dpi=600)

            # Close plot
            plt.close()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the table
    df_ = df.copy()

    # Drop failed cases
    df_.dropna(subset=["fitStates"], inplace=True)

    # Check for midpoint results
    if "midPoint" not in df_.columns:
        raise ValueError("Results do not include window midpoint. Older plotting tool may be required for these results.")

    # Bodges to allow plotting with older results
    if "fitDates" not in df_.columns:
        df_.rename(columns={"dates": "fitDates"}, inplace=True)
    if "referencePropagator" not in df_.columns:
        df_["referencePropagator"] = "SLR"
    if "name" not in df_.columns:
        df_["name"] = df_["sp3name"].apply(lambda x: SP3MAP[x])

    # Calculate fit end
    # TODO: rename start/duration
    df_["fitEnd"] = df_["midPoint"] + df_["duration"] / 2

    # Calculate date relative to epoch
    func = lambda dates, epoch: duration_to_days(dates - epoch)
    func_dates = lambda x: func(x.fitDates, x.fitEnd)
    func_testDates = lambda x: func(x.testDates, x.fitEnd)
    df_["daysPostFitEnd"] = df_.apply(func_dates, axis=1)
    df_["testDaysPostFitEnd"] = df_.apply(func_testDates, axis=1)

    # Calculate position RMSE
    df_["fitErrorRMS"] = df_["fitError"].apply(lambda x: np.sqrt(np.mean(x**2)))

    # Calculate error difference
    df_["errorDiff"] = df_["fitError"] - df_["sampleError"]
    df_["errorDiffBetter"] = df_["errorDiff"].apply(lambda x: (x < 0.0).astype(float))

    # Return updated table
    return df_


if __name__ == "__main__":
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("fname", type=str)
    parser_args = parser.parse_args()

    # Extract filename
    fname = parser_args.fname

    # Load results
    df = pd.read_pickle(fname)

    # Preprocess results
    df = preprocess(df)

    # Calculate number of window and sample sizes
    nwindow = len(np.unique(df["duration"]))
    nsample = len(np.unique(df["samples"]))

    # Check window and sample size compatibility
    # TODO: consider merged tables
    if (nwindow > 1) and (nsample == 1):
        # Plot window mesh
        plot_window_mesh(df, fname)
    elif (nwindow == 1) and (nsample > 1):
        # Plot sample mesh
        plot_sample_mesh(df, fname)
    else:
        print("Incompatible number of windows and samples")

    # Plot proportions
    plot_proportion(df, fname)

    # Plot error histories
    plot_errors(df, fname)
