# Standard imports
from datetime import datetime

# External imports
from brent.download import SEMDownloader


def main(output: str) -> None:
    # Get years
    start = 1990
    end = datetime.now().year

    # Download SEM files
    SEMDownloader.download(start, end, output)
