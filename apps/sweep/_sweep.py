# Standard import
from dataclasses import asdict
from datetime import datetime, timedelta
import json
import os
import signal
import sys

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# External imports
from brent.bias import BiasFactory
from brent.covariance import CovarianceFactory
from brent.filter import OrekitBatchLeastSquares, ThalassaBatchLeastSquares
from brent.frames import RTN
from brent.io import Saver
from brent.propagators import (
    Propagator,
    NumericalPropagatorParameters,
    SP3Propagator,
    TLEPropagator,
)
from brent.util import generate_parameter_permutations


def load(fpath):
    # Load configuration file
    with open(fpath, "r") as fid:
        arguments_raw = json.load(fid)

    # Extract time parameters
    times = arguments_raw["times"]
    dateformat = "%Y-%m-%d"
    start = datetime.strptime(times["start"], dateformat)
    end = datetime.strptime(times["end"], dateformat)
    period = f"{times['period']}D"
    testDuration = timedelta(times["testDuration"])
    fitDuration = [timedelta(iduration) for iduration in times["fitDuration"]]
    fitSamples = times["fitSamples"]

    # Extract propagator
    propagator = arguments_raw["propagator"]
    if propagator not in ["orekit", "thalassa"]:
        raise ValueError("Unknown proapgator")

    # Extract default model
    model = arguments_raw["model"]

    # Extract noise model
    noise = arguments_raw["noise"]

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
                pd.date_range(start, end, freq=period),
                pd.DatetimeIndex([end]),
            ),
        ),
    )

    # Store arguments
    arguments = {
        "midPoint": start_,
        "fitDuration": fitDuration,
        "fitSamples": fitSamples,
        "propagator": [propagator],
        "noise": [noise],
        "testDuration": [testDuration],
    }

    # Return spacecraft and arguments
    return spacecraft, arguments


def fit(spacecraft, parameters):
    # Extract number of samples
    fitSamples = parameters["fitSamples"]

    ## Extract dates
    # Fit
    fitDuration = parameters["fitDuration"]
    fitMidPoint = parameters["midPoint"]
    fitStartDate = fitMidPoint - fitDuration / 2
    fitEndDate = fitMidPoint + fitDuration / 2
    # Test
    testStartDate = fitStartDate
    testEndDate = fitEndDate + parameters["testDuration"]

    # Generate dates
    # TODO: set sampling technique, testing duration
    fitDates = pd.date_range(fitStartDate, fitEndDate, periods=fitSamples)
    testDates = pd.date_range(testStartDate, testEndDate, freq="1h")

    # Extract physical model parameters
    model = NumericalPropagatorParameters(**spacecraft["model"])

    # Extract bias model
    bias = spacecraft["bias"]
    biasModel = BiasFactory.create(**bias)

    # Extract sample noise model
    noise = parameters["noise"]
    covarianceProvider = CovarianceFactory.create(**noise)

    # Load TLEs
    # TODO: include TLEs from before window?
    samplePropagator = TLEPropagator.load(
        spacecraft["tle"],
        fitStartDate,
        fitEndDate,
    )

    # Generate psuedo-observation states
    sampleStates = samplePropagator.propagate(fitDates)

    # Debias the sample states
    sampleStates = biasModel.debias(fitDates, sampleStates)

    # Create filter
    propagator = parameters["propagator"]
    if propagator == "orekit":
        filter = OrekitBatchLeastSquares(
            fitDates,
            sampleStates,
            model,
            covarianceProvider,
        )
    elif propagator == "thalassa":
        filter = ThalassaBatchLeastSquares(
            fitDates,
            sampleStates,
            model,
            covarianceProvider,
        )
    else:
        raise ValueError(f"Unknown propagator: {propagator}")

    # Execute filter
    fitPropagator = filter.estimate()

    # Get estimated covariance
    fitCovariance = filter.getEstimatedCovariance()[0:6, 0:6]

    # Get estimated model
    modelEstimated = filter.getEstimatedModel()

    # Generate fit states
    fitStates = fitPropagator.propagate(fitDates)

    # Calculate RTN transformations
    rtn = RTN.getTransform(fitStates)

    # Transform fit covariance to RTN
    fitCovarianceRTN = rtn[0, :, :] @ fitCovariance @ rtn[0, :, :].T

    # Calculate state residuals
    # TODO: check order
    deltaStates = sampleStates - fitStates

    # Transform state residuals to RTN
    deltaStatesRTN = RTN.transform(rtn, deltaStates)

    # Calculate residual sample covariances
    residualCovariance = np.cov(deltaStates, rowvar=False)
    residualCovarianceRTN = np.cov(deltaStatesRTN, rowvar=False)

    # Extract test propagator
    if spacecraft["sp3propagator"] is None:
        testPropagator = TLEPropagator.load(spacecraft["tle"])
        referencePropagator = "TLE"
    else:
        testPropagator: Propagator = spacecraft["sp3propagator"]
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
        # Fit results
        "fitStates": fitStates,
        "modelEstimated": asdict(modelEstimated),
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


def main(input: str, output_dir: str) -> None:
    # Declare results saver
    saver = Saver(output_dir)
    saver.save_input(input)

    # Get main process identifier
    pid = os.getpid()

    def exit_handler(signum, frame) -> None:
        # Ignore signals from child processes
        if os.getpid() != pid:
            return

        # Save results
        # TODO: handle repeated signals while saver is still saving
        saver.save(final=True)

        # Exit process
        sys.exit()

    # Register exit handler
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, exit_handler)

    # Load arguments
    spacecraft, arguments = load(input)

    # Load SP3 propagators
    for ispacecraft in tqdm(spacecraft, desc="SP3 load"):
        # No SP3 propagator
        if ispacecraft["sp3"] is None:
            ispacecraft["sp3propagator"] = None
            continue

        # Load SP3 propagator
        ispacecraft["sp3propagator"] = SP3Propagator.load(
            ispacecraft["sp3"],
            ispacecraft["sp3name"],
        )

    # Generate parameter permutations
    argument_permutations = generate_parameter_permutations(arguments)

    # Generate input pairs
    input_pairs = [
        (ispacecraft, iarguments)
        for iarguments in argument_permutations
        for ispacecraft in spacecraft
    ]

    def sort_key(x):
        # Extract arguments
        arguments = x[1]

        # Sort by duration, then number of samples
        return arguments["fitDuration"], arguments["fitSamples"]

    # Sort by descending expected execution time
    input_pairs = sorted(input_pairs, key=sort_key, reverse=True)

    # Iterate through input pairs
    for arg in tqdm(input_pairs, desc="Fitting", dynamic_ncols=True):
        # Execute fit
        fit = fit_wrapper(arg)

        # Store fit result
        saver.update(fit)

    # Save final results
    saver.save(final=True)

    # TODO: delete checkpoint files?
