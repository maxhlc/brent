# Standard imports
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import os
import signal
import sys
from typing import List

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
    ) -> None:
        # Declare state
        self.state = SaverState.DEINITIALISED

        # Loop until directory initialised
        ntries = 0
        while self.state != SaverState.INITIALISED:
            # Save start time and name
            self.time = datetime.now(timezone.utc)
            self.name = self.time.strftime("%Y%m%d_%H%M%S")

            # Create directory
            initialised = self._create_directory(root, self.name)

            # Set state
            if initialised:
                self.state = SaverState.INITIALISED

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

        # Save initial metadata
        self.save_metadata()

        # Set state to running
        self.state = SaverState.RUNNING

    def update(self, result: dict) -> None:
        # Append result to results
        self.results.append(result)

        # Trigger auto-save
        if len(self.results) % self.save_period == 0:
            self.save()

    def save(self, final: bool = False) -> None:
        # Check that saver is still running
        if self.state != SaverState.RUNNING:
            raise RuntimeError("Saver not running")

        # Set state
        if final:
            self.state = SaverState.FINAL

        # Set filename suffix
        suffix = "" if final else f"_{self.checkpoint}"

        # Generate results filepath
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

        # Save metadata
        self.save_metadata()

        # Set state to finished
        if self.state == SaverState.FINAL:
            self.state = SaverState.FINISHED

    def save_metadata(self) -> None:
        # Generate metadata filepath
        fname = os.path.join(self.directory, self.name + "_metadata.txt")

        # Declare lines list
        lines: List[str] = []

        # Check run state
        if self.state == SaverState.INITIALISED:
            # Start metadata
            lines = self._initial_metadata()
        elif self.state == SaverState.RUNNING:
            # TODO: save iteration progress?
            pass
        elif self.state == SaverState.FINAL:
            # Final metadata
            lines = self._final_metdata()
        else:
            # Raise error
            raise RuntimeError("Incompatible run state")

        # Ensure lines have linebreak
        lines = [line + "\n" for line in lines]

        # Save lines to metdata file
        # TODO: error handling?
        with open(fname, "a") as fp:
            fp.writelines(lines)

    def _initial_metadata(self) -> List[str]:
        # Declare lines list
        lines: List[str] = []

        # Git commit
        git_hash = brent.util.get_commit()
        lines.append(f"Git commit: {git_hash}")

        # Start time
        time = self.time
        lines.append(f"Start time: {time.isoformat()}")

        # Return lines
        return lines

    def _final_metdata(self) -> List[str]:
        # Declare lines list
        lines: List[str] = []

        # End time
        time = datetime.now(timezone.utc)
        lines.append(f"End time:   {time.isoformat()}")

        # Execution time
        exectime = (time - self.time).total_seconds()
        lines.append(f"Exec. time: {exectime:.2f} s")

        # Number of fits
        nfits = len(self.results)
        lines.append(f"No. fits:   {nfits}")

        # Fit period
        fitperiod = exectime / nfits
        lines.append(f"Fit rate:   {fitperiod:.2f} s/fit")

        # Return lines
        return lines

    def _create_directory(self, root: str, name: str) -> bool:
        # Return if already initialised
        if self.state != SaverState.DEINITIALISED:
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

        # Update checkpoint state
        self.checkpoint = "checkpoint_b" if self.niter % 2 == 0 else "checkpoint_a"


class SaverState(Enum):
    DEINITIALISED = 1
    INITIALISED = 2
    RUNNING = 3
    FINAL = 4
    FINISHED = 5


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
    model = brent.propagators.NumericalPropagatorParameters(**spacecraft["model"])

    # Extract bias model
    bias = spacecraft["bias"]
    biasModel = brent.bias.BiasFactory.create(**bias)

    # Extract sample noise model
    noise = parameters["noise"]
    covarianceProvider = brent.covariance.CovarianceFactory.create(**noise)

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


if __name__ == "__main__":
    # Parse input
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./input/sweep.json")
    parser.add_argument("-o", "--output_dir", type=str, default="./output/")
    parser_args = parser.parse_args()

    # Execute
    main(parser_args.input, parser_args.output_dir)
