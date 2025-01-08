# Future imports
from __future__ import annotations

# Standard imports
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import datetime
import json
import multiprocessing as mp
import os.path
from string import Template
from typing import List, Tuple

# Third-party imports
import pandas as pd
from tqdm import tqdm

# External imports
from brent.util import CDDISDownloader

# Internal imports
from .application import Application, ApplicationFactory


@dataclass
class Parameters:
    # CDDIS authentication parameters
    username: str
    password: str

    # Path templates
    filename_template: str
    url_root_template: str

    # Providers
    providers: List[str]

    # Object name
    name: str

    # Dates
    start: datetime
    end: datetime
    frequency: str

    # Output directory
    output_path: str

    @staticmethod
    def load(fname: str) -> Parameters:
        # Load parameters from file
        with open(fname, "r") as fid:
            parameters = json.load(fid)

        # Format dates
        dateformat = "%Y-%m-%d"
        parameters["start"] = datetime.strptime(parameters["start"], dateformat)
        parameters["end"] = datetime.strptime(parameters["end"], dateformat)
        parameters["frequency"] = f"{parameters['frequency']}D"

        # Return parameters object
        return Parameters(**parameters)


@dataclass
class Filepaths:
    # Store filepaths
    url: str
    output_filepath: str


class FilepathGenerator:

    def __init__(
        self,
        filename_template: str,
        url_root_template: str,
        output_path: str,
    ) -> None:
        # Store path templates
        self.filename_template = Template(filename_template)
        self.url_root_template = Template(url_root_template)

        # Store output path
        self.output_path = output_path

    def generate_filepath(self, name: str, provider: str, date: str) -> Filepaths:
        # Generate filepaths
        filename = self.filename_template.substitute(
            name=name,
            provider=provider,
            date=date,
        )
        url_root = self.url_root_template.substitute(
            name=name,
            provider=provider,
            date=date,
        )

        # Create URL and output filepath
        url = os.path.join(url_root, filename)
        output_filepath = os.path.join(self.output_path, filename)

        # Return filepaths object
        return Filepaths(url, output_filepath)


@dataclass
class DownloadWorkerBundle:
    # Store parameters for a download worker
    downloader: CDDISDownloader
    filepaths: Filepaths


def download_worker(bundle: DownloadWorkerBundle) -> Tuple[DownloadWorkerBundle, bool]:
    # Unpack bundle
    downloader = bundle.downloader
    filepaths = bundle.filepaths

    try:
        # Try to download file
        downloader.download(filepaths.url, filepaths.output_filepath)

        # Return success
        return bundle, True
    except:
        # Return failure
        return bundle, False


def main(input: str) -> None:
    # Load parameters
    parameters = Parameters.load(input)

    # Generate dates
    dates = [
        date.strftime("%y%m%d")
        for date in pd.date_range(
            start=parameters.start,
            end=parameters.end,
            freq=parameters.frequency,
        )
    ]

    # Create filepath generator
    filepathgenerator = FilepathGenerator(
        parameters.filename_template,
        parameters.url_root_template,
        parameters.output_path,
    )

    # Generate filepaths
    filepaths = [
        filepathgenerator.generate_filepath(parameters.name, provider, date)
        for date in dates
        for provider in parameters.providers
    ]

    # Create file downloader
    downloader = CDDISDownloader(parameters.username, parameters.password)

    # Create worker bundles
    bundles = [DownloadWorkerBundle(downloader, filepath) for filepath in filepaths]

    # Spawn multiprocessed loop to download files
    with mp.Pool() as pool, tqdm(total=len(filepaths)) as pbar:
        # Iterate through bundles
        for bundle, sucess in pool.imap_unordered(download_worker, bundles):
            # Print error if download unsuccessful
            if not sucess:
                tqdm.write(f"Failed: {bundle.filepaths.url}")

            # Update progress bar
            pbar.update()


@ApplicationFactory.register("cddis", "Download SP3 files from CDDIS")
class CDDIS(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Extract arguments
        input = arguments.input

        # Execute download
        main(input)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument(
            "-i",
            "--input",
            type=str,
            default="./input/download.json",
            help="Input filepath",
        )
