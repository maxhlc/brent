# External imports
import brent


def main(input: str, output: str) -> None:
    # Load ANTEX file
    antex = brent.io.AntexFile.load(input)

    # Get PRNs
    prns = antex.get_prn()

    # Save PRNs
    prns.to_csv(
        output,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S.%f",
    )
