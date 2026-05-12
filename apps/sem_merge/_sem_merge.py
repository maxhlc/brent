# Standard imports
from datetime import datetime, timedelta
from glob import glob
import os

# Third-party imports
import numpy as np
import pandas as pd
from tqdm import tqdm

# External imports
from brent.io.sem import SEMFile


def load(path: str) -> pd.DataFrame:
    # Find SEM files
    files = glob(path, recursive=True)

    # Declare list for SEM files
    sems: list[SEMFile] = []

    # Iterate through files
    for file in tqdm(files, desc="Loading SEM files"):
        # Extract year from directory
        # NOTE: assumes file directories labelled by year
        components = file.split(os.sep)
        year = int(components[-2])

        # Load SEM file
        sem = SEMFile.load(file, year)

        # Store SEM file
        sems.append(sem)

    # Convert SEM files to DataFrame
    df = pd.concat([sem.to_dataframe() for sem in sems])

    # Trim columns
    df = df[["svn", "prn", "year", "week", "seconds"]].copy()

    # Return SEM records
    return df


def gps_to_datetime(year: int, week: int, seconds: int) -> datetime:
    # TODO: deal with difference between UTC and GPS time
    # TODO: timedelta ignores leapseconds, resulting in discrepancy

    # Declare constants
    WEEK_TO_SECONDS = 7.0 * 86400.0
    ROLLOVER_SECONDS = 1024 * WEEK_TO_SECONDS

    # Declare GPS epoch
    GPS_EPOCH = datetime(1980, 1, 6)

    # Calculate start-of-year dates
    year_epoch = datetime(year, 1, 1)
    year_epoch_seconds = (year_epoch - GPS_EPOCH).total_seconds()
    year_mod_seconds = year_epoch_seconds % ROLLOVER_SECONDS

    # Calculate almanac date in seconds
    mod_seconds = week * WEEK_TO_SECONDS + seconds

    # Calculate difference between almanac date and start-of-year date
    # NOTE: wrapped to catch roll-over
    delta_seconds = (mod_seconds - year_mod_seconds) % ROLLOVER_SECONDS

    # Return datetime
    return year_epoch + timedelta(seconds=delta_seconds)


def merge_date_ranges(grp: pd.DataFrame) -> pd.DataFrame:
    # Sort by start date
    grp.sort_values("start", inplace=True)

    # Merge if preceeding end date matches following start date
    # Solution from: https://stackoverflow.com/a/44269511 (accessed 2025-04-24)
    dt_groups = (grp["start"] != grp["end"].shift()).cumsum()
    return grp.groupby(dt_groups).agg({"start": "first", "end": "last"})


def main(input: str, output: str) -> None:
    # Load SEM files
    df = load(input)

    # Get start dates
    # TODO: calculate once for each SEM instead of each record?
    df["start"] = df.apply(
        lambda x: gps_to_datetime(int(x["year"]), int(x["week"]), int(x["seconds"])),
        axis=1,
    )

    # Find unique start dates
    start_dates = np.unique(df["start"])

    # Create start-to-end map
    date_pairs = list(zip(start_dates[:-1], start_dates[1:]))
    date_pairs.append((start_dates[-1], start_dates[-1]))
    date_map = dict(date_pairs)

    # Set end dates
    df["end"] = df["start"].map(date_map)

    # Merge consecutive records
    df2 = (
        df.groupby(["svn", "prn"])
        .apply(merge_date_ranges)
        .reset_index()
        .drop("level_2", axis=1)
    )

    # Store SVN/PRN mapping
    df2.to_csv(output, index=False)
