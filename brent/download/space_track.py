# Standard imports
from datetime import datetime
import json
import os.path
import requests
from typing import Dict, List


class SpaceTrackDownloader:

    def __init__(self, username: str, password: str) -> None:
        # Store credentials
        self.credentials = {"identity": username, "password": password}

        # Open session
        self.session = requests.Session()

        # Set authentication flag
        self.authenticated = False

    def __del__(self) -> None:
        # Close session
        self.session.close()

    def authenticate(self) -> None:
        # Try to authenticate
        response = self.session.post(self.URL_AUTH, self.credentials)

        # Raise error if authentication fails
        if response.status_code != 200:
            raise RuntimeError("Space-Track authentication failed")

        # Set authentication flag
        self.authenticated = True

    def download(
        self,
        identifiers: List[str],
        start: datetime,
        end: datetime,
        output_directory: None | str = None,
    ) -> List[List[Dict[str, str]]]:
        # Ensure dates are consistent
        if end <= start:
            raise RuntimeError("End must later than start")

        # Construct query
        # NOTE: times are discarded from the start and end
        identifiers_str = ",".join(identifiers)
        start_str = start.strftime(self.URL_DATE_FORMAT)
        end_str = end.strftime(self.URL_DATE_FORMAT)
        query = os.path.join(
            self.URL_BASE,
            "basicspacedata/query",
            "class/gp_history/",
            f"NORAD_CAT_ID/{identifiers_str}",
            "orderby/EPOCH%20ASC",
            f"EPOCH/{start_str}--{end_str}",
            "format/json",
        )

        # Authenticate if not authenticated
        if not self.authenticated:
            self.authenticate()

        # Query Space-Track
        response = self.session.get(query)

        # Throw error if request fails
        if response.status_code != 200:
            raise RuntimeError("Space-Track query failed")

        # Deserialise response content
        content: List[Dict[str, str]] = json.loads(response.content)

        # Split into seperate element sets
        elsets = [
            [icontent for icontent in content if icontent["NORAD_CAT_ID"] == identifier]
            for identifier in identifiers
        ]

        # Check if output directory provided
        if output_directory is not None:
            # Iterate through identifiers
            for identifier, ielsets in zip(identifiers, elsets):
                # Generate filepath
                filepath = os.path.join(output_directory, f"{identifier}.json")

                # Ensure file directory exists
                os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

                # Save element sets
                with open(filepath, "w") as fp:
                    json.dump(ielsets, fp, indent=4)

        # Return element sets
        return elsets

    # URLs
    URL_BASE = "https://www.space-track.org/"
    URL_AUTH = os.path.join(URL_BASE, "ajaxauth/login")

    # Formats
    URL_DATE_FORMAT = "%Y-%m-%d"
