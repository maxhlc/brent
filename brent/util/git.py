# Standard imports
import subprocess

# Internal imports
import brent.paths


def get_commit() -> str:
    # NOTE: Solution adapted from https://stackoverflow.com/a/21901260 (accessed 2024-11-20)
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=brent.paths.ROOT_DIR,
        )
        .decode("ascii")
        .strip()
    )
