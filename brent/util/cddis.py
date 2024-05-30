# Standard imports
import os

# Third-party imports
import requests


class CDDISDownloader:

    def __init__(self, username: str, password: str) -> None:
        # Store authentication parameters
        self.username = username
        self.password = password

        # Open session
        self.session = requests.Session()

        # Set authentication parameters
        self.session.auth = (self.username, self.password)

    def __del__(self):
        # Close session
        self.session.close()

    def download(self, url: str, filepath: str, timeout: float = 10) -> None:
        # Request file (repeated for authentication)
        # Solution from: https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python (accessed 2024-05-23)
        response_ = self.session.request("get", url, timeout=timeout)
        response = self.session.get(response_.url, timeout=timeout)

        # Raise any errors
        response.raise_for_status()

        # Ensure file directory exists
        os.makedirs(
            os.path.dirname(os.path.abspath(filepath)),
            exist_ok=True,
        )

        with open(filepath, "wb") as fid:
            # Save file
            fid.write(response.content)
