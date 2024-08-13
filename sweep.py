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
            "tle": ispacecraft["tle"],
            "sp3": ispacecraft["sp3"],
            "sp3name": ispacecraft["sp3name"],
            "output": ispacecraft["output"],
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

    # Declare propagator builder
    fitPropagatorBuilder = brent.propagators.NumericalPropagator.builder(
        fitDates[0],
        sampleStates[0, :],
        model,
    )

    # Generate observations
    fitObservations = brent.filter.generate_observations(
        fitDates,
        sampleStates,
        covarianceProvider,
    )

    # Create filter
    filter = brent.filter.OrekitBatchLeastSquares(fitPropagatorBuilder, fitObservations)

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


def main(spacecraft, arguments):
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

    # Execute fits
    fits = [fit_wrapper(arg) for arg in tqdm(input_pairs, desc="Fit exec")]

    # Create DataFrame of results
    df = pd.DataFrame(fits)

    # Set output directory, and ensure it exists
    directory = "./output/"
    os.makedirs(os.path.abspath(directory), exist_ok=True)

    # Save results
    # TODO: use different format than Pickle
    ia_max = 3
    for ia in range(ia_max + 1):
        # Set suffix (used if retry limit exceeded)
        if ia == ia_max:
            # Print warning message
            print("Reverting to UUID mode")

            # Generate suffix
            suffix = uuid.uuid4().hex
        else:
            # Set blank suffix
            suffix = ""

        # Generate file name
        now = datetime.now(timezone.utc)
        fname = directory + now.strftime("%Y%m%d_%H%M%S_%f") + suffix + ".pkl"

        try:
            # Try to save the results
            with open(fname, "xb") as fp:
                df.to_pickle(fp)

            # Break retry loop
            break
        except:
            # Print error message
            print(f"Error saving file at: {fname} (Attempt {ia + 1}/{ia_max})")


if __name__ == "__main__":
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("input", type=str, default="./input/sweep.json")
    parser_args = parser.parse_args()

    # Load arguments
    spacecraft, arguments = load(parser_args.input)

    # Execute
    main(spacecraft, arguments)
