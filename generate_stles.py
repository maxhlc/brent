# Standard imports
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from typing import List, Dict

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Orekit imports
import orekit
from orekit.pyhelpers import absolutedate_to_datetime
from org.orekit.propagation.analytical.tle import TLE

# Internal imports
import brent


@dataclass
class Parameters:
    # SP3 parameters
    sp3path: str
    sp3id: str

    # TLE parameters
    tlepath: str

    # Fit parameters
    duration: float
    bstar: bool

    # Metadata
    stlename: str

    # Output
    outputpath: str


def generate_dates(
    epochs: List[datetime],
    duration: float,
    freq: str = "120min",
) -> np.ndarray:
    # Set start and end dates
    start = epochs[0]
    end = epochs[-1] + timedelta(duration)

    # Generate dates
    dates = pd.date_range(start, end, freq=freq).to_pydatetime()

    # Insert TLE epochs into dates
    dates = np.append(dates, np.array(epochs))
    dates.sort()

    # Return dates
    return dates


def tle_to_stle(
    dates: np.ndarray,
    states: np.ndarray,
    epoch: datetime,
    duration: float,
    bstar: bool,
) -> TLE:
    # Determine forward/backward fit
    if duration > 0:
        start = epoch
        end = epoch + timedelta(duration)
        reference_index = 0
    else:
        start = epoch - timedelta(duration)
        end = epoch
        reference_index = -1

    # Extract observations
    idx = np.logical_and(dates >= start, dates <= end)
    dates_ = dates[idx]
    states_ = states[idx, :]

    # Generate S-TLE
    filter = brent.filter.SyntheticTLEGenerator(
        dates_,
        states_,
        reference_index,
        bstar,
    )
    fit = filter.estimate()

    # Extract S-TLE
    stle = fit.propagators[0][1].propagator.getTLE()

    # Return S-TLE
    return stle


def tle_stle_delta(tle: TLE, stle: TLE) -> np.ndarray:
    # Convert (S-)TLEs to vector
    tle_ = brent.filter.SyntheticTLEGenerator._tle_to_vector(tle, True)
    stle_ = brent.filter.SyntheticTLEGenerator._tle_to_vector(stle, True)

    # Calculate difference
    delta = stle_ - tle_

    # Add argument of latitude differences
    u = delta[4] + delta[5]
    delta = np.append(delta, u)

    # Return difference
    return delta


def tles_to_stles(
    dates: np.ndarray,
    states: np.ndarray,
    epochs: List[datetime],
    duration: float,
    bstar: bool,
) -> List[TLE]:
    # Return S-TLEs
    return [
        tle_to_stle(dates, states, epoch, duration, bstar) for epoch in tqdm(epochs)
    ]


def plot_delta(epochs: List[datetime], delta: np.ndarray) -> None:
    # Set axis labels
    yaxlabel = [
        r"$\Delta n$",
        r"$\Delta e$",
        r"$\Delta i$",
        r"$\Delta \Omega$",
        r"$\Delta \omega$",
        r"$\Delta M$",
        r"$\Delta B^*$",
        r"$\Delta u$",
    ]

    # Calculate number of variables
    nvar = delta.shape[1]

    # Create subplots
    fig, axes = plt.subplots(nvar, 1, sharex=True)

    # Iterate through variables and plot
    for idx in range(nvar):
        axes[idx].plot(epochs, delta[:, idx])
        axes[idx].set_ylabel(yaxlabel[idx])


def save(
    path: str,
    tles: List[TLE],
    object_name: str = "SYNTHETIC",
    object_id: str = "0000-000A",
    extra: Dict[str, str | float | int | bool] = {},
) -> None:
    # Get current datetime
    now = datetime.now()

    # Generate records from TLEs
    records = [
        {
            "OBJECT_NAME": object_name,
            "OBJECT_ID": object_id,
            "EPOCH": absolutedate_to_datetime(tle.getDate()).isoformat(),
            "CREATION_DATE": now.isoformat(),
            "TLE_LINE1": tle.getLine1(),
            "TLE_LINE2": tle.getLine2(),
            "BRENT_EXTRA": extra,
        }
        for tle in tles
    ]

    # Save to file
    with open(path, "w") as fid:
        json.dump(records, fid, indent=4)


def main(parameters: Parameters) -> None:
    # Import SP3
    sp3 = brent.propagators.SP3Propagator.load(parameters.sp3path, parameters.sp3id)

    # Import actual TLEs
    tles = brent.propagators.TLEPropagator._load(parameters.tlepath)

    # Extract epochs
    epochs = [absolutedate_to_datetime(tle.getDate()) for tle in tles]

    # Generate dates
    dates = generate_dates(epochs, parameters.duration)

    # Propagate SP3 states
    statesSP3 = sp3.propagate(dates)

    # Generate S-TLEs
    stles = tles_to_stles(
        dates,
        statesSP3,
        epochs,
        parameters.duration,
        parameters.bstar,
    )

    # Save S-TLEs
    fname = f"./output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_stles"
    save(
        fname + ".json",
        stles,
        object_name=parameters.stlename,
        extra={
            "DURATION": parameters.duration,
            "BSTAR": parameters.bstar,
        },
    )

    # Calculate TLE differences
    delta = np.array([tle_stle_delta(tle, stle) for tle, stle in zip(tles, stles)])

    # Create (S-)TLE propagators
    tlePropagator = brent.propagators.TLEPropagator(tles)
    stlePropagator = brent.propagators.TLEPropagator(stles)

    # Propagate (S-)TLE states
    statesTLE = tlePropagator.propagate(dates)
    statesSTLE = stlePropagator.propagate(dates)

    # Calculate position magnitudes
    rTLE = np.linalg.norm(statesTLE[:, 0:3], axis=1)
    rSTLE = np.linalg.norm(statesSTLE[:, 0:3], axis=1)

    # Calculate state errors
    deltaStatesTLE = statesTLE - statesSP3
    deltaStatesSTLE = statesSTLE - statesSP3

    # Calculate RTN transformations
    RTN = brent.frames.rtn(statesSP3)

    # Calculate RTN errors
    deltaStatesTLERTN = np.einsum("ijk,ik -> ij", RTN, deltaStatesTLE)
    deltaStatesSTLERTN = np.einsum("ijk,ik -> ij", RTN, deltaStatesSTLE)

    # Plot along-track angular errors
    plt.figure()
    plt.plot(dates, deltaStatesTLERTN[:, 1] / rTLE, label="TLE")
    plt.plot(dates, deltaStatesSTLERTN[:, 1] / rSTLE, label="S-TLE")
    plt.legend()
    plt.xlabel("Date [-]")
    plt.ylabel(r"$\Delta \theta_T$ [rad]")

    # Plot position error magnitude
    plt.figure()
    plt.plot(dates, np.linalg.norm(deltaStatesTLE[:, 0:3], axis=1), label="TLE")
    plt.plot(dates, np.linalg.norm(deltaStatesSTLE[:, 0:3], axis=1), label="S-TLE")
    plt.legend()
    plt.xlabel("Date [-]")
    plt.ylabel(r"$\Delta r$ [m]")

    # Plot TLE RTN errors
    plt.figure()
    plt.plot(dates, deltaStatesTLERTN[:, 0:3], label=["R", "T", "N"])
    plt.legend()
    plt.xlabel("Date [-]")
    plt.ylabel(r"$\Delta r$ [m]")

    # Plot S-TLE RTN errors
    plt.figure()
    plt.plot(dates, deltaStatesSTLERTN[:, 0:3], label=["R", "T", "N"])
    plt.legend()
    plt.xlabel("Date [-]")
    plt.ylabel(r"$\Delta r$ [m]")

    # Plot (S)-TLE differences
    plot_delta(epochs, delta)

    # Show results
    plt.show()


if __name__ == "__main__":
    # Set parameters
    parameters = Parameters(
        **{
            "sp3path": "./data/sp3/etalon2/*.sp3",
            "sp3id": "L54",
            "tlepath": "./data/tle/20026.json",
            "stlename": "Etalon 2 (SYNTHETIC)",
            "duration": 3,
            "bstar": False,
            "outputpath": "./output/stle/",
        }
    )

    # Execute main function
    main(parameters)
