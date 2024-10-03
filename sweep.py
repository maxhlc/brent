# Standard imports
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# Internal imports
import brent


class Saver:

    def __init__(
        self,
        root: str,
        save_period: int = 50,
        retry_limit: int = 5,
    ):
        # Declare initialisation
        self.initialised = False

        # Loop until directory initialised
        ntries = 0
        while not self.initialised:
            # Save start time and name
            self.time = datetime.now(timezone.utc)
            self.name = self.time.strftime("%Y%m%d_%H%M%S")

            # Create directory
            self.initialised = self._create_directory(root, self.name)

            # Increment number of tries
            ntries += 1

            # Throw error if not initialised
            if ntries > retry_limit:
                raise ValueError("Failed to initialise")

        # Store save period
        self.save_period = save_period

        # Initialise state
        self.niter = 0
        self._increment_state()

        # Results
        self.results = []

    def update(self, result: dict) -> None:
        # Append result to results
        self.results.append(result)

        # Trigger auto-save
        if len(self.results) % self.save_period == 0:
            self.save()

    def save(self, final: bool = False) -> None:
        # Set filename suffix
        suffix = "" if final else f"_{self.state}"

        # Generate file name
        fname = os.path.join(self.directory, self.name + suffix + ".pkl")

        # Convert results to DataFrame
        df = pd.DataFrame(self.results)

        # Save results
        try:
            # Try to save the results
            with open(fname, "wb") as fp:
                df.to_pickle(fp)

            # Update state
            if not final:
                self._increment_state()
        except:
            # Print error message
            tqdm.write(f"Error saving file at: {fname}")

    def _create_directory(self, root: str, name: str) -> bool:
        # Return if already initialised
        if self.initialised:
            return False

        try:
            # Generate output directory
            directory = os.path.abspath(os.path.join(root, name))

            # Throw error if the directory already exists
            os.makedirs(directory, exist_ok=False)

            # Store directory
            self.directory = directory
            self.initialised = True

            # Return success
            return True
        except Exception as e:
            tqdm.write(str(e))

            # Return failure
            return False

    def save_input(self, input: str) -> None:
        # Generate output path
        output = os.path.join(self.directory, f"{self.name}_input.json")

        # Copy the input file to the output path
        with open(input, "rb") as src, open(output, "wb") as dest:
            dest.write(src.read())

    def _increment_state(self) -> None:
        # Increment number iteration number
        self.niter += 1

        # Update state
        self.state = "checkpoint_b" if self.niter % 2 == 0 else "checkpoint_a"


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
                pd.date_range(start, end, freq=frequency),
                pd.DatetimeIndex([end]),
            ),
        ),
    )

    # Store arguments
    arguments = {
        "midPoint": start_,
        "duration": duration,
        "samples": samples,
        "propagator": [propagator],
        "noise": [noise],
    }

    # Return spacecraft and arguments
    return spacecraft, arguments


def fit(spacecraft, parameters):
    # Extract number of samples
    samples = parameters["samples"]

    ## Extract dates
    # Fit
    fitDuration = parameters["duration"]
    fitMidPoint = parameters["midPoint"]
    fitStartDate = fitMidPoint - fitDuration / 2
    fitEndDate = fitMidPoint + fitDuration / 2
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
    propagator = parameters["propagator"]
    if propagator == "orekit":
        filter = brent.filter.OrekitBatchLeastSquares(
            fitDates,
            sampleStates,
            model,
            covarianceProvider,
        )
    elif propagator == "thalassa":
        filter = brent.filter.ThalassaBatchLeastSquares(
            fitDates,
            sampleStates,
            model,
            covarianceProvider,
        )

    # Execute filter
    fitPropagator = filter.estimate()

    # Get estimated covariance
    fitCovariance = filter.getCovariance()[0:6, 0:6]

    # Get estimated model
    modelEstimated = filter.getModel()

    # Generate fit states
    fitStates = fitPropagator.propagate(fitDates)

    # Calculate RTN transformations
    rtn = brent.frames.RTN.getTransform(fitStates)

    # Transform fit covariance to RTN
    fitCovarianceRTN = rtn[0, :, :] @ fitCovariance @ rtn[0, :, :].T

    # Calculate state residuals
    # TODO: check order
    deltaStates = sampleStates - fitStates

    # Transform state residuals to RTN
    deltaStatesRTN = brent.frames.RTN.transform(rtn, deltaStates)

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

    # Load arguments
    spacecraft, arguments = load(input)

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

    def sort_key(x):
        # Extract arguments
        arguments = x[1]

        # Sort by duration, then number of samples
        return arguments["duration"], arguments["samples"]

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


if __name__ == "__main__":
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./input/sweep.json")
    parser.add_argument("-o", "--output_dir", type=str, default="./output/")
    parser_args = parser.parse_args()

    # Execute
    main(parser_args.input, parser_args.output_dir)
