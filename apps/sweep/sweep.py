# Standard import
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("sweep", "Sweep multiple filter cases")
class Sweep(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._sweep import main

        # Extract arguments
        input = arguments.input
        output_dir = arguments.output_dir

        # Execute sweep
        main(input, output_dir)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument(
            "-i",
            "--input",
            type=str,
            default="./input/sweep.json",
            help="Input filepath",
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            default="./output/",
            help="Output directory",
        )
