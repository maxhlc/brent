# Standard imports
from argparse import ArgumentParser
from datetime import datetime

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set output figure size
FIGSIZE = (5.5, 4.0)

# Declare function to convert duration to days
duration_to_days = np.vectorize(lambda x: x / np.timedelta64(1, "D"))


def plot_window_mesh(df: pd.DataFrame, fname: str) -> None:
    # Iterate through objects
    for object in np.unique(df["sp3name"]):
        # Extract corresponding sub-table for object
        df_object = df[df["sp3name"] == object].copy()

        # Extract x-y-z variables
        x = df_object["fitEpoch"].to_numpy()
        y = duration_to_days(df_object["duration"])
        z = df_object["fitErrorRMS"].to_numpy()

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
        # TODO: dynamic
        plt.xlim((datetime(2022, 1, 1), datetime(2023, 1, 1)))

        # Set axis labels
        plt.xlabel("Fit Epoch [-]")
        plt.ylabel("Fit Window Size [days]")

        # Add colour bar
        plt.colorbar(label="Position RMSE [m]")

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
    # Iterate through objects
    for object in np.unique(df["sp3name"]):
        # Extract corresponding sub-table for object
        df_object = df[df["sp3name"] == object].copy()

        # Extract x-y-z variables
        x = df_object["fitEpoch"].to_numpy()
        y = df_object["samples"].to_numpy()
        z = df_object["fitErrorRMS"].to_numpy()

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
        # TODO: dynamic
        plt.xlim((datetime(2022, 1, 1), datetime(2023, 1, 1)))

        # Set axis labels
        plt.xlabel("Fit Epoch [-]")
        plt.ylabel("Sample Size [-]")

        # Add colour bar
        plt.colorbar(label="Position RMSE [m]")

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


def plot_errors(df: pd.DataFrame, window: np.timedelta64, fname: str) -> None:
    # Calculate number of days in window
    days = int(window / np.timedelta64(1, "D"))

    # Extract subtable
    df_window = df[df["duration"] == window].copy()

    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot position RMSEs
    sns.lineplot(data=df_window, x="fitEpoch", y="fitErrorRMS", hue="sp3name")

    # Set axis labels
    plt.xlabel("Fit Epoch [-]")
    plt.ylabel("Position RMSE [m]")

    # Update legend title
    plt.legend(title="SP3 ID")

    # Set limits
    # TODO: dynamic
    plt.xlim((datetime(2022, 1, 1), datetime(2023, 1, 1)))
    plt.ylim((0, 10e3))

    # Format dates
    fig.autofmt_xdate()

    # Add grid
    plt.grid()

    # Set layout
    plt.tight_layout()

    # Export plot
    plt.savefig(f"{fname}_rmse_{days}D.png", dpi=600)

    # Close plot
    plt.close()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the table
    df_ = df.copy()

    # Calculate fit epoch
    df_["fitEpoch"] = df_["start"] + df_["duration"]

    # Calculate position RMSE
    df_["fitErrorRMS"] = df_["fitError"].apply(lambda x: np.sqrt(np.mean(x**2)))

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

    # Plot error histories
    for window in np.unique(df["duration"]):
        plot_errors(df, window, fname)
