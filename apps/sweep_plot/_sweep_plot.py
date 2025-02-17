# Standard imports
from copy import deepcopy
from math import ceil

# Third-party imports
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns

# Store pre-execution Matplotlib parameters
RCPARAMS = deepcopy(plt.rcParams)

# Set output figure size
FIGSIZE = (3.25, 3.25)
FIGSIZE_LARGE = (6.6, 8.4)

# Set font size
plt.rcParams.update({"font.size": 7})

# Declare name map
SP3MAP = {
    None: "Unassigned",  # TODO: replace bodge for objects without SP3 name
    "L51": "LAGEOS-1",
    "L52": "LAGEOS-2",
    "L53": "Etalon 1",
    "L54": "Etalon 2",
}

LIMITS = {
    # Calibration
    "LAGEOS-1": 8,
    "LAGEOS-2": 8,
    "Etalon 1": 8,
    "Etalon 2": 8,
    # Test
    "Navstar 1": 40,
    "Navstar 2": 40,
    "Astra 1E": 100,
    "Astra 1H": 100,
}

INTERVALS = {
    # Calibration
    "LAGEOS-1": 1,
    "LAGEOS-2": 1,
    "Etalon 1": 1,
    "Etalon 2": 1,
    # Test
    "Navstar 1": 5,
    "Navstar 2": 5,
    "Astra 1E": 10,
    "Astra 1H": 10,
}

THRESHOLDS = {
    # Calibration
    "LAGEOS-1": 0.25,
    "LAGEOS-2": 0.75,
    "Etalon 1": 1,
    "Etalon 2": 1.5,
    # Test
    "Navstar 1": 5,
    "Navstar 2": 20,
    "Astra 1E": 20,
    "Astra 1H": 20,
}

# Declare function to convert duration to days
duration_to_days = np.vectorize(lambda x: x / np.timedelta64(1, "D"))


def roundup(val: float, interval: float) -> float:
    # Return value, rounded up to specified interval
    return val + (interval - val % interval)


def plot_window_mesh(df: pd.DataFrame, fname: str) -> None:
    # Calculate x-limits
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
        y = duration_to_days(df_object["fitDuration"])
        z = df_object["fitErrorRMS"].to_numpy() / 1000.0

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

        # Calculate z-limits
        zupper = LIMITS[object]
        zn = 2 * int(zupper // INTERVALS[object]) + 1
        zlevels = np.linspace(0.0, zupper, zn)

        # Configure colour map
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_over(cmap(1.0))

        # Extend colour map if maximum value exceeded
        extend = "max" if np.nanmax(z) > zupper else "neither"

        # Plot mesh
        plt.contourf(
            x[idx],
            y[idx],
            z[idx],
            zlevels,
            norm=colors.Normalize(0, zupper, clip=True),
            cmap=cmap,
            extend=extend,
        )

        # Set limits
        # TODO: y-axis
        plt.xlim(xlim)

        # Set axis labels
        plt.xlabel("Fit Window Midpoint [-]")
        plt.ylabel("Fit Window Size [days]")

        # Add colour bar
        cbar = plt.colorbar(
            label=f"Position RMSE (w.r.t. {referencePropagator}) [km]",
            extend=extend,
            aspect=40,
        )
        cbar.set_ticks(
            [
                INTERVALS[object] * n
                for n in range(0, int(zupper // INTERVALS[object]) + 1)
            ]
        )

        # Plot threshold
        cbar.ax.plot([0, 1], [THRESHOLDS[object]] * 2, "r")

        # Format dates
        fig.autofmt_xdate()

        # Add grid
        plt.grid(linewidth=0.25)

        # Set layout
        plt.tight_layout()

        # Export plot
        plt.savefig(f"{fname}_duration_rmse_{object}.png", dpi=600)

        # Close plot
        plt.close()

def plot_window_mesh_combined(df: pd.DataFrame, fname: str) -> None:
    # Get unique objects (preserving order from limits list)
    # TODO: make generic
    unique_objects = np.unique(df["name"])
    objects = [name for name in LIMITS if name in unique_objects]

    # Create subplots
    fig, axes = plt.subplots(
        nrows=ceil(len(objects) / 2),
        ncols=2,
        sharex=True,
        sharey=True,
        figsize=FIGSIZE_LARGE,
        constrained_layout=True,
    )

    # Calculate x-limits
    # TODO: cleaner implementation
    xlim = (
        np.min(df["midPoint"]),
        np.max(df["midPoint"]),
    )

    # Iterate through objects
    for ax, object in zip(axes.ravel(), objects):
        # Extract corresponding sub-table for object
        df_object = df[df["name"] == object].copy()

        # Extract x-y-z variables
        x = df_object["midPoint"].to_numpy()
        y = duration_to_days(df_object["fitDuration"])
        z = df_object["fitErrorRMS"].to_numpy() / 1000.0

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

        # Calculate z-limits
        zupper = LIMITS[object]
        zn = 2 * int(zupper // INTERVALS[object]) + 1
        zlevels = np.linspace(0.0, zupper, zn)

        # Configure colour map
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_over(cmap(1.0))

        # Extend colour map if maximum value exceeded
        extend = "max" if np.nanmax(z) > zupper else "neither"

        # Plot mesh
        ctf = ax.contourf(
            x[idx],
            y[idx],
            z[idx],
            zlevels,
            norm=colors.Normalize(0, zupper, clip=True),
            cmap=cmap,
            extend=extend,
        )

        # Set limits
        # TODO: y-axis
        ax.set_xlim(xlim)

        # Set title
        ax.set_title(object, loc="right")

        # Add colour bar
        cbar = plt.colorbar(
            ctf,
            label=f"Position RMSE (w.r.t. {referencePropagator}) [km]",
            extend=extend,
            aspect=40,
            ax=ax,
        )
        cbar.set_ticks(
            [
                INTERVALS[object] * n
                for n in range(0, int(zupper // INTERVALS[object]) + 1)
            ]
        )

        # Plot threshold
        cbar.ax.plot([0, 1], [THRESHOLDS[object]] * 2, "r")

        # Format dates
        fig.autofmt_xdate()

        # Add grid
        ax.grid(linewidth=0.25)

    # Set axis labels
    for ax in axes[-1, :]:
        ax.set_xlabel("Fit Window Midpoint [-]")

    for ax in axes[:, 0]:
        ax.set_ylabel("Fit Window Size [days]")

    # Export plot
    plt.savefig(f"{fname}_duration_rmse_mesh.png", dpi=600)

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
        y = df_object["fitSamples"].to_numpy()
        z = df_object["fitErrorRMS"].to_numpy() / 1000.0

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
        # TODO: y-axis
        plt.xlim(xlim)

        # Set axis labels
        plt.xlabel("Fit Window Midpoint [-]")
        plt.ylabel("Sample Size [-]")

        # Add colour bar
        # TODO: configure same as plot_window_mesh
        plt.colorbar(label=f"Position RMSE (w.r.t. {referencePropagator}) [km]")

        # Format dates
        fig.autofmt_xdate()

        # Add grid
        plt.grid(linewidth=0.25)

        # Set layout
        plt.tight_layout()

        # Export plot
        plt.savefig(f"{fname}_samples_rmse_{object}.png", dpi=600)

        # Close plot
        plt.close()


def plot_proportion(df: pd.DataFrame, fname: str) -> None:
    # Iterate through windows
    for window in np.unique(df["fitDuration"]):
        # Iterate through samples
        for samples in np.unique(df["fitSamples"]):
            # Calculate number of days in window
            days = int(window / np.timedelta64(1, "D"))

            # Extract subtable
            idx = np.logical_and(
                df["fitDuration"] == window,
                df["fitSamples"] == samples,
            )
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
    ymax = np.nanmax(df["fitErrorRMS"])
    interval = 10 ** (np.floor(np.log10(ymax)) + 1) / 5
    yupper = roundup(ymax, interval)
    ylim = (0, yupper)

    # Iterate through windows
    for window in np.unique(df["fitDuration"]):
        # Iterate through samples
        for samples in np.unique(df["fitSamples"]):
            # Calculate number of days in window
            days = int(window / np.timedelta64(1, "D"))

            # Extract subtable
            df_ = df[df["fitDuration"] == window].copy()

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


def plot_window_mean(df: pd.DataFrame, fname: str) -> None:
    # Preprocess
    df_ = df.copy()
    df_["fitDuration"] = duration_to_days(df_["fitDuration"])
    df_["fitErrorRMS"] /= 1e3

    # Group results
    groups = ["name", "fitDuration"]
    columns = [*groups, "fitErrorRMS"]
    df_grouped = df_[columns].groupby(groups, sort=False).mean()

    # Save results
    df_grouped.to_csv(f"{fname}_mean.csv")

    # Set upper limit
    # TODO: replace with automatic limits
    ymax = np.max(df_grouped["fitErrorRMS"])
    top = 6 if ymax < 10 else roundup(ymax, 10.0)

    # Create plot
    plt.figure(figsize=FIGSIZE)

    # Plot mean RMSEs
    sns.lineplot(
        data=df_,
        x="fitDuration",
        y="fitErrorRMS",
        errorbar=None,
        hue="name",
    )

    # Set limits
    plt.xlim((np.min(df_["fitDuration"]), np.max(df_["fitDuration"])))
    plt.ylim((0, top))

    # Set locator
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Set labels
    plt.xlabel("Fit Window Size [days]")
    plt.ylabel("Mean Position RMSE [km]")

    # Update legend title
    plt.legend(title="Satellite")

    # Set layout
    plt.tight_layout()

    # Add grid
    plt.grid()

    # Export plot
    plt.savefig(f"{fname}_duration_mean_rmse.png", dpi=600)
    plt.savefig(f"{fname}_duration_mean_rmse.pdf")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy of the table
    df_ = df.copy()

    # Drop failed cases
    df_.dropna(subset=["fitStates"], inplace=True)

    # Check for midpoint results
    if "midPoint" not in df_.columns:
        raise ValueError(
            "Results do not include window midpoint."
            + " "
            + "Older plotting tool may be required for these results."
        )

    # Bodges to allow plotting with older results
    if "fitDates" not in df_.columns:
        df_.rename(columns={"dates": "fitDates"}, inplace=True)
    if "referencePropagator" not in df_.columns:
        df_["referencePropagator"] = "SLR"
    if "name" not in df_.columns:
        df_["name"] = df_["sp3name"].apply(lambda x: SP3MAP[x])
    if "fitDuration" not in df_.columns:
        df_.rename(columns={"duration": "fitDuration"}, inplace=True)
    if "fitSamples" not in df_.columns:
        df_.rename(columns={"samples": "fitSamples"}, inplace=True)

    # Calculate fit end
    # TODO: rename start/duration
    df_["fitEnd"] = df_["midPoint"] + df_["fitDuration"] / 2

    # Calculate date relative to epoch
    func = lambda dates, epoch: duration_to_days(dates - epoch)
    func_dates = lambda x: func(x.fitDates, x.fitEnd)
    func_testDates = lambda x: func(x.testDates, x.fitEnd)
    df_["daysPostFitEnd"] = df_.apply(func_dates, axis=1)
    df_["testDaysPostFitEnd"] = df_.apply(func_testDates, axis=1)

    # Calculate position RMSE
    def rmse_func(series):
        # Extract window end and test dates
        fitEnd = series["fitEnd"]
        testDates = series["testDates"]

        # Extract post-fit indices
        idx = testDates >= fitEnd

        # Extract fit position errors
        fitError = series["fitError"]

        # Return post-fit position RMSE
        return np.sqrt(np.mean(fitError[idx] ** 2))

    df_["fitErrorRMS"] = df_.apply(rmse_func, axis=1)

    # Calculate error difference
    df_["errorDiff"] = df_["fitError"] - df_["sampleError"]
    df_["errorDiffBetter"] = df_["errorDiff"].apply(lambda x: (x < 0.0).astype(float))

    # Return updated table
    return df_


def main(input: str) -> None:
    # Load results
    df = pd.read_pickle(input)

    # Preprocess results
    df = preprocess(df)

    # Calculate number of window and sample sizes
    nwindow = len(np.unique(df["fitDuration"]))
    nsample = len(np.unique(df["fitSamples"]))

    # Check window and sample size compatibility
    # TODO: consider merged tables
    if (nwindow > 1) and (nsample == 1):
        # Plot window mean
        plot_window_mean(df, input)

        # Plot window mesh
        plot_window_mesh(df, input)

        # Plot combined window mesh
        plot_window_mesh_combined(df, input)
    elif (nwindow == 1) and (nsample > 1):
        # Plot sample mesh
        plot_sample_mesh(df, input)
    else:
        print("Incompatible number of windows and samples")

    # Plot proportions
    plot_proportion(df, input)

    # Plot error histories
    plot_errors(df, input)

    # Reset changes to Matplotlib parameters
    # TODO: not executed in cases of crash etc., switch to context manager
    plt.rcParams.update(RCPARAMS)
