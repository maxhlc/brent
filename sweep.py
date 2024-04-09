# Standard imports
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
import json

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

    # Store arguments
    arguments = {
        "start": pd.date_range(start, end, freq=frequency),
        "duration": duration,
        "samples": samples,
        "noise": noise,
    }

    # Return spacecraft and arguments
    return spacecraft, arguments


def fit(spacecraft, parameters):
    # Extract dates
    fitStartDate = parameters["start"]
    fitDuration = parameters["duration"]
    fitEndDate = fitStartDate + fitDuration

    # Extract physical model parameters
    model = brent.propagators.ModelParameters(**spacecraft["model"])

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

    # Extract sample and test dates
    # TODO: set sampling technique, testing duration
    samples = parameters["samples"]
    dates = pd.date_range(fitStartDate, fitEndDate, periods=samples)
    testDates = pd.date_range(fitStartDate, fitEndDate + timedelta(30), freq="1H")

    # Extract test propagator
    testPropagator = spacecraft["sp3propagator"]

    # Load TLEs
    tlePropagator = brent.io.load_tle_propagator(
        spacecraft["tle"], np.min(dates), np.max(dates)
    )

    # Generate psuedo-observation states
    sampleStates = tlePropagator.propagate(dates)

    # Debias the sample states
    sampleStates = biasModel.debias(dates, sampleStates)

    # Create filter
    filter = brent.filter.OrekitBatchLeastSquares(
        dates, sampleStates, model, covarianceProvider
    )

    # Execute filter
    fitPropagator = filter.estimate()

    # Get estimated covariance
    fitCovariance = filter.covariance()[0:6, 0:6]

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

    # Calculate test states
    sampleTestStates = tlePropagator.propagate(testDates)
    fitTestStates = fitPropagator.propagate(testDates)
    testStates = testPropagator.propagate(testDates)

    # Calculate position error
    sampleError = np.linalg.norm(sampleTestStates - testStates, axis=1)
    fitError = np.linalg.norm(fitTestStates - testStates, axis=1)

    # Calculate proportion of fit period where the fit outperforms the samples
    proportion = np.mean(fitError <= sampleError)

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
        ispacecraft["sp3propagator"] = brent.io.load_sp3_propagator(
            ispacecraft["sp3"], ispacecraft["sp3name"]
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

    # Save results
    now = datetime.now(timezone.utc)
    fname = "./output/" + now.strftime("%Y%m%d_%H%M%S") + ".pkl"
    df.to_pickle(fname)


if __name__ == "__main__":
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, default="./input/sweep.json")
    parser_args = parser.parse_args()

    # Load arguments
    spacecraft, arguments = load(parser_args.input)

    # Execute
    main(spacecraft, arguments)
