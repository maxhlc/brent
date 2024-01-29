# Standard imports
from datetime import datetime, timedelta, timezone

# Third party imports
import pandas as pd
from tqdm import tqdm
import timeout_decorator


# Internal imports
import brent
from main import main as fit


def worker(arg):
    try:
        # Create arguments dataclass
        args_ = brent.io.Arguments(**arg, verbose=False, plot=False)

        # Wrap fit method with timeout
        @timeout_decorator.timeout(360)
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
    # Define default common model parameters
    default_model = {
        "cr": 0.0,
        "potential": True,
        "potential_degree": 10,
        "potential_order": 10,
        "moon": True,
        "sun": True,
        "srp": True,
        "srp_estimate": True,
    }

    # Define arguments
    args_ = {
        "start": pd.date_range(datetime(2022, 1, 1), datetime(2022, 11, 20), freq="3D"),
        "duration": [timedelta(10)],
        "spacecraft": [
            {
                "tle": "./data/tle/8820.json",
                "sp3": "./data/sp3/lageos1/*.sp3",
                "sp3name": "L51",
                "output": "./output/LAGEOS1",
                "model": brent.propagators.ModelParameters(
                    mass=406.965,
                    area_srp=0.282743338,
                    **default_model
                ),
            },
            {
                "tle": "./data/tle/22195.json",
                "sp3": "./data/sp3/lageos2/*.sp3",
                "sp3name": "L52",
                "output": "./output/LAGEOS2",
                "model": brent.propagators.ModelParameters(
                    mass=405.38,
                    area_srp=0.282743338,
                    **default_model
                ),
            },
            {
                "tle": "./data/tle/19751.json",
                "sp3": "./data/sp3/etalon1/*.sp3",
                "sp3name": "L53",
                "output": "./output/ETALON1",
                "model": brent.propagators.ModelParameters(
                    mass=1415.0,
                    area_srp=1.315098959,
                    **default_model
                ),
            },
            {
                "tle": "./data/tle/20026.json",
                "sp3": "./data/sp3/etalon2/*.sp3",
                "sp3name": "L54",
                "output": "./output/ETALON2",
                "model": brent.propagators.ModelParameters(
                    mass=1415.0,
                    area_srp=1.315098959,
                    **default_model
                ),
            },
        ],
    }

    # Generate argument perturbations
    args = brent.util.generate_parameter_permutations(args_)

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
