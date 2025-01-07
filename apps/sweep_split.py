# Standard imports
from argparse import ArgumentParser, Namespace
import copy
import json

# Internal imports
from .application import Application, ApplicationFactory


def main(input: str) -> None:
    # Load sweep configuration
    with open(input) as fp:
        config = json.load(fp)

    # Split by object
    configs = []
    for ispacecraft in config["spacecraft"]:
        config_ = copy.deepcopy(config)
        config_["spacecraft"] = [ispacecraft]
        configs.append(config_)

    # Save separated configurations
    for idx, iconfig in enumerate(configs):
        # Generate filename
        output_path = input + f"_split_{idx:03d}.json"

        # Save configuration
        with open(output_path, "w") as fp:
            json.dump(iconfig, fp, indent=4)


@ApplicationFactory.register("sweep_split")
class SweepSplit(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Extract arguments
        input = arguments.input

        # Execute sweep split
        main(input)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument("input", type=str)
