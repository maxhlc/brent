# Future imports
from __future__ import annotations

# Standard imports
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os.path
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

    @staticmethod
    def load(fname: str) -> Parameters:
        # Load parameters from file
        with open(fname, "r") as fid:
            return Parameters(**json.load(fid))

    def save(self, fname: str) -> None:
        # Save parameters to file
        with open(fname, "w") as fid:
            json.dump(asdict(self), fid, indent=4)


def generate_dates(
    epochs: List[datetime],
    duration: float,
    freq: str = "120min",
) -> np.ndarray:
    # Determine forward/backward fit and set dates
    if duration > 0:
        start = epochs[0]
        end = epochs[-1] + timedelta(duration)
    else:
        start = epochs[0] + timedelta(duration)
        end = epochs[-1]

    # Generate dates
    dates = pd.date_range(start, end, freq=freq).to_pydatetime()

    # Insert TLE epochs into dates
    dates = np.append(dates, np.array(epochs))
    dates.sort()

    # Return dates
    return dates


def states_to_stle(
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
        start = epoch + timedelta(duration)
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


def tle_delta(tle1: TLE, tle2: TLE) -> np.ndarray:
    # Convert TLEs to vectors
    tle1_ = brent.filter.SyntheticTLEGenerator._tle_to_vector(tle1, True)
    tle2_ = brent.filter.SyntheticTLEGenerator._tle_to_vector(tle2, True)

    # Calculate difference
    delta = tle2_ - tle1_

    # Calculate difference in argument of latitude
    # NOTE: this definiton uses mean anomaly, instead of
    #       the common definition with true anomaly
    du = delta[4] + delta[5]

    # Append argument of latitude difference to deltas
    delta = np.append(delta, du)

    # Wrap angles
    idx = [
        2,  # Inclination
        3,  # Right ascension of the ascending node
        4,  # Argument of periapsis
        5,  # Mean anomaly
        7,  # Arugment of latitude
    ]
    delta[idx] = np.arctan2(np.sin(delta[idx]), np.cos(delta[idx]))

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
        states_to_stle(dates, states, epoch, duration, bstar) for epoch in tqdm(epochs)
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


def save_stles(
    path: str,
    tles: List[TLE],
    object_name: str = "SYNTHETIC",
    object_id: str = "0000-000A",
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
        }
        for tle in tles
    ]

    # Save to file
    with open(path, "w") as fid:
        json.dump(records, fid, indent=4)


def save(parameters: Parameters, tles: List[TLE], stles: List[TLE]):
    # Ensure directory exists
    os.makedirs(os.path.abspath(parameters.outputpath), exist_ok=True)

    # Generate filepaths
    date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_parameters = os.path.join(
        parameters.outputpath,
        f"{date_string}_parameters.json",
    )
    fname_tles = os.path.join(
        parameters.outputpath,
        f"{date_string}_tles.json",
    )
    fname_stles = os.path.join(
        parameters.outputpath,
        f"{date_string}_stles.json",
    )

    # Save parameters
    parameters.save(fname_parameters)

    # Save copy of TLEs
    # TODO: store in memory, instead of file copy after sweep?
    with open(parameters.tlepath, "rb") as fin, open(fname_tles, "wb") as fout:
        fout.write(fin.read())

    # Save S-TLEs
    save_stles(fname_stles, stles, object_name=parameters.stlename)


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

    # Save results
    save(parameters, tles, stles)

    # Calculate TLE differences
    delta = np.array([tle_delta(tle, stle) for tle, stle in zip(tles, stles)])

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
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="./input/stle.json")
    parser_args = parser.parse_args()

    # Load parameters
    parameters = Parameters.load(parser_args.input)

    # Execute main function
    main(parameters)
