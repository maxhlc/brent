# Standard imports
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
import itertools

# Third party imports
import pandas as pd
from tqdm import tqdm
import timeout_decorator


def generate_parameter_permutations(parameters):
    # Extract keys and values
    keys, values = zip(*parameters.items())

    # Return generated permutations
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


# Internal imports
from main import main as fit


@dataclass
class Arguments:
    start: datetime
    duration: timedelta
    tle: str
    sp3: str
    sp3name: str
    output: str
    plot: bool = False
    verbose: bool = False


def worker(arg):
    try:
        # Create argument dataclass
        args_ = Arguments(
            arg["start"],
            arg["duration"],
            arg["tle"],
            arg["sp3"],
            arg["sp3name"],
            arg["output"],
        )

        # Wrap fit method with timeout
        @timeout_decorator.timeout(60)
        def timeout_fit(args):
            return fit(args)

        # Return arguments with fit results
        return {**arg, **timeout_fit(args_)}
    except Exception as e:
        # Print error
        print(e)

        # Return arguments
        return {**arg}


def main():
    # Define arguments
    args_ = {
        "start": pd.date_range(datetime(2022, 1, 1), datetime(2022, 1, 7), freq="1D"),
        "duration": [timedelta(7)],
        "spacecraft": [
            {
                "tle": "./data/tle/8820.json",
                "sp3": "./data/sp3/lageos1/*.sp3",
                "sp3name": "L51",
                "output": "./output/LAGEOS1",
            },
            {
                "tle": "./data/tle/22195.json",
                "sp3": "./data/sp3/lageos2/*.sp3",
                "sp3name": "L52",
                "output": "./output/LAGEOS2",
            },
            {
                "tle": "./data/tle/19751.json",
                "sp3": "./data/sp3/etalon1/*.sp3",
                "sp3name": "L53",
                "output": "./output/ETALON1",
            },
            {
                "tle": "./data/tle/20026.json",
                "sp3": "./data/sp3/etalon2/*.sp3",
                "sp3name": "L54",
                "output": "./output/ETALON2",
            },
        ],
    }

    # Generate argument perturbations
    args = generate_parameter_permutations(args_)

    # Expand spacecraft arguments
    args = [{**arg, **arg["spacecraft"]} for arg in args]
    for arg in args:
        del arg["spacecraft"]

    # Execute fits
    fits = [worker(arg) for arg in tqdm(args)]

    # Create DataFrame of results
    df = pd.DataFrame(fits)

    # Save results
    now = datetime.now(timezone.utc)
    fname = "./output/" + now.strftime("%Y%m%d_%H%M%S") + ".pkl"
    df.to_pickle(fname)


if __name__ == "__main__":
    # Execute main function
    main()
