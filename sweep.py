# Standard imports
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import json
import os
import uuid

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Internal imports
import brent


def load(fpath):
    # Load configuration file
    with open(fpath, "r") as fid:
        arguments_raw = json.load(fid)

    # Extract time parameters
    times = arguments_raw["times"]
    dateformat = "%Y-%m-%d"
    start = datetime.strptime(times["start"], dateformat)
    end = datetime.strptime(times["end"], dateformat)
    frequency = f"{times['frequency']}D"
    duration = [timedelta(iduration) for iduration in times["duration"]]
    samples = times["samples"]

    # Extract default model
    model = arguments_raw["model"]

    # Extract noise model
    noise = [arguments_raw["noise"]]

    # Extract spacecraft parameters
    spacecraft = [
        {
            "name": ispacecraft["name"],
            "tle": ispacecraft["tle"],
            "sp3": ispacecraft["sp3"],
            "sp3name": ispacecraft["sp3name"],
            "model": {**model, **ispacecraft["model"]},
            "bias": ispacecraft["bias"],
        }
        for ispacecraft in arguments_raw["spacecraft"]
    ]

    # Generate fit window dates (ensuring end date is included)
    start_ = pd.DatetimeIndex(
        np.unique(
            np.append(
                pd.date_range(start, end, freq=frequency),
                pd.DatetimeIndex([end]),
            ),
        ),
    )

    # Store arguments
    arguments = {
        "start": start_,
        "duration": duration,
        "samples": samples,
        "noise": noise,
    }

    # Return spacecraft and arguments
    return spacecraft, arguments


def fit(spacecraft, parameters):
    # Extract number of samples
    samples = parameters["samples"]

    ## Extract dates
    # Fit
    fitStartDate = parameters["start"]
    fitDuration = parameters["duration"]
    fitEndDate = fitStartDate + fitDuration
    # Test
    testStartDate = fitStartDate
    testDuration = fitDuration + timedelta(30)
    testEndDate = testStartDate + testDuration

    # Generate dates
    # TODO: set sampling technique, testing duration
    fitDates = pd.date_range(fitStartDate, fitEndDate, periods=samples)
    testDates = pd.date_range(testStartDate, testEndDate, freq="1h")

    # Extract physical model parameters
    model = brent.propagators.NumericalPropagatorParameters(**spacecraft["model"])

    # Extract bias model
    bias = spacecraft["bias"]
    if bias["model"] == "none":
        biasModel = brent.bias.BiasModel()
    elif bias["model"] == "simplifiedalongtracksinusoidal":
        biasModel = brent.bias.SimplifiedAlongtrackSinusoidal(*bias["parameters"])
    else:
        raise ValueError("Unknown bias model")

    # Extract sample noise model
    noise = parameters["noise"]
    if noise["frame"] == "rtn":
        covarianceProvider = brent.filter.RTNCovarianceProvider(np.array(noise["std"]))
    else:
        covarianceProvider = brent.filter.CovarianceProvider()

    # Load TLEs
    # TODO: include TLEs from before window?
    samplePropagator = brent.propagators.TLEPropagator.load(
        spacecraft["tle"],
        fitStartDate,
        fitEndDate,
    )

    # Generate psuedo-observation states
    sampleStates = samplePropagator.propagate(fitDates)

    # Debias the sample states
    sampleStates = biasModel.debias(fitDates, sampleStates)

    # Create filter
    filter = brent.filter.OrekitBatchLeastSquares(
        fitDates,
        sampleStates,
        model,
        covarianceProvider,
    )

    # Execute filter
    fitPropagator = filter.estimate()

    # Get estimated covariance
    fitCovariance = filter.covariance()[0:6, 0:6]

    # Generate fit states
    fitStates = fitPropagator.propagate(fitDates)

    # TODO: extract updated model parameters (e.g. SRP, if estimated)

    # Calculate RTN transformations
    RTN = brent.frames.rtn(fitStates)

    # Transform fit covariance to RTN
    fitCovarianceRTN = RTN[0, :, :] @ fitCovariance @ RTN[0, :, :].T

    # Calculate state residuals
    # TODO: check order
    deltaStates = sampleStates - fitStates

    # Transform state residuals to RTN
    deltaStatesRTN = np.einsum("ijk,ik -> ij", RTN, deltaStates)

    # Calculate residual sample covariances
    residualCovariance = np.cov(deltaStates, rowvar=False)
    residualCovarianceRTN = np.cov(deltaStatesRTN, rowvar=False)

    # Extract test propagator
    if spacecraft["sp3propagator"] is None:
        testPropagator = brent.propagators.TLEPropagator.load(spacecraft["tle"])
        referencePropagator = "TLE"
    else:
        testPropagator: brent.propagators.Propagator = spacecraft["sp3propagator"]
        referencePropagator = "SLR"

    # Calculate test states
    testStates = testPropagator.propagate(testDates)
    sampleTestStates = samplePropagator.propagate(testDates)
    fitTestStates = fitPropagator.propagate(testDates)

    # Calculate position errors
    sampleError = np.linalg.norm(sampleTestStates[:, 0:3] - testStates[:, 0:3], axis=1)
    fitError = np.linalg.norm(fitTestStates[:, 0:3] - testStates[:, 0:3], axis=1)

    # Calculate proportion of fit period where the fit outperforms the samples
    proportion = np.mean(fitError <= sampleError)

    # Return results
    return {
        # Fit parameters
        "fitDates": fitDates,
        "sampleStates": sampleStates,
        "fitStates": fitStates,
        # Fit metrics
        "deltaStates": deltaStates,
        "deltaStatesRTN": deltaStatesRTN,
        "fitCovariance": fitCovariance,
        "fitCovarianceRTN": fitCovarianceRTN,
        "residualCovariance": residualCovariance,
        "residualCovarianceRTN": residualCovarianceRTN,
        # Test parameters
        "referencePropagator": referencePropagator,
        "testDates": testDates,
        "testStates": testStates,
        "sampleTestStates": sampleTestStates,
        "fitTestStates": fitTestStates,
        # Test metrics
        "sampleError": sampleError,
        "fitError": fitError,
        "proportion": proportion,
    }


def fit_wrapper(inputs):
    # Split the inputs
    spacecraft, parameters = inputs

    # Declare result dictionary with inputs
    result = {**spacecraft, **parameters}

    # Try to fit
    try:
        result.update(**fit(spacecraft, parameters))
    except Exception as e:
        tqdm.write(str(e))

    # Remove objects which cannot be pickled
    del result["sp3propagator"]

    # Return result
    return result


def save(df: pd.DataFrame, time: datetime, state: str) -> None:
    # Stringfy time
    time_str = time.strftime("%Y%m%d_%H%M%S_%f")

    # Set output directory, and ensure it exists
    directory = os.path.abspath(os.path.join("output", time_str))

    # Ensure that the output directory exists
    os.makedirs(directory, exist_ok=True)

    # TODO: use different format than Pickle

    # Set filename suffix
    suffix = "" if state == "" else f"_{state}"

    # Generate file name
    fname = os.path.join(directory, time_str + suffix + ".pkl")

    # Save results
    try:
        # Try to save the results
        with open(fname, "wb") as fp:
            df.to_pickle(fp)
    except:
        # Print error message
        tqdm.write(f"Error saving file at: {fname}")


def main(spacecraft, arguments):
    # Save start time
    start = datetime.now(timezone.utc)

    # Load SP3 propagators
    for ispacecraft in tqdm(spacecraft, desc="SP3 load"):
        # No SP3 propagator
        if ispacecraft["sp3"] is None:
            ispacecraft["sp3propagator"] = None
            continue

        # Load SP3 propagator
        ispacecraft["sp3propagator"] = brent.propagators.SP3Propagator.load(
            ispacecraft["sp3"],
            ispacecraft["sp3name"],
        )

    # Generate parameter permutations
    argument_permutations = brent.util.generate_parameter_permutations(arguments)

    # Generate input pairs
    input_pairs = [
        (ispacecraft, iarguments)
        for iarguments in argument_permutations
        for ispacecraft in spacecraft
    ]

    # Declare list of results
    fits = []

    # Set checkpoint state
    # NOTE: this is to rotate checkpoint files to prevent corruption
    #       if a crash occurs during the checkpoint save
    state = "checkpoint_a"

    # Set checkpoint frequency
    # NOTE: at least twice, or every 50 iterations
    checkpoint_frequency = np.min((len(input_pairs) // 3, 50))

    for idx, arg in enumerate(tqdm(input_pairs, desc="Fit exec")):
        # Execute fit
        fit = fit_wrapper(arg)

        # Store fit result
        fits.append(fit)

        # Checkpoint
        # TODO: decide on number of cases? time since last checkpoint?
        if ((idx + 1) % checkpoint_frequency == 0) and (idx + 1 != len(input_pairs)):
            # Create DataFrame of current results
            df = pd.DataFrame(fits)

            # Save current results
            save(df, start, state)

            # Set state for next checkpoint
            state = "checkpoint_b" if state == "checkpoint_a" else "checkpoint_a"

    # Create DataFrame of final results
    df = pd.DataFrame(fits)

    # Save results
    save(df, start, "")

    # TODO: delete checkpoint files?


if __name__ == "__main__":
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("input", type=str, default="./input/sweep.json")
    parser_args = parser.parse_args()

    # Load arguments
    spacecraft, arguments = load(parser_args.input)

    # Execute
    main(spacecraft, arguments)
