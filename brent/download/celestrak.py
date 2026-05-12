# Standard imports
from dataclasses import dataclass
import os.path
import re

# Third-party imports
import requests
from tqdm import tqdm


@dataclass
class SEMLink:
    year: int
    path: str
    url: str


class SEMDownloader:

    @classmethod
    def get_urls(cls, years: list[int]) -> list[SEMLink]:
        # Create session
        session = requests.Session()

        # Declare list of links
        links: list[SEMLink] = []

        # Iterate through years
        for year in tqdm(years, "Finding SEM paths"):
            # Generate URL
            url = os.path.join(cls.URL_BASE, f"GPS/almanac/SEM/{year}/")

            # Get page
            page = session.get(url)

            # Find paths
            paths: list[str] = re.findall(cls.PATH_PATTERN, page.content.decode())

            # Iterate through paths
            for path in paths:
                # Generate link
                link = SEMLink(year, path, os.path.join(cls.URL_BASE, path.lstrip("/")))

                # Store link
                links.append(link)

        # Return links
        return links

    @classmethod
    def download(cls, start: int, end: int, output_directory: str) -> None:
        # Ensure dates are consistent
        if end < start:
            raise RuntimeError("End must later than start")

        # Get links
        links = cls.get_urls(list(range(start, end + 1)))

        # Create session
        session = requests.Session()

        # Iterate through links
        # TODO: parallel downloads?
        for link in tqdm(links, desc="Downloading SEM files"):
            # Extract year and filename
            year = link.year
            filename = os.path.basename(link.path)

            # Generate filepath
            filepath = os.path.join(output_directory, str(year), filename)

            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

            # Skip if file already exists
            if os.path.exists(filepath):
                continue

            # Download file
            # TODO: error handling
            with open(filepath, "wb") as fp:
                # Download file
                response = session.get(link.url)

                # Save contents
                fp.write(response.content)

    # URL root
    URL_BASE: str = "https://celestrak.org/"

    # SEM path pattern
    PATH_PATTERN: str = r"/GPS/almanac/SEM/[0-9]{4,4}/almanac\.sem\.week[0-9]{4,4}\.[0-9]{6,6}\.txt"
