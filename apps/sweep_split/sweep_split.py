# Standard imports
from argparse import ArgumentParser, Namespace


# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("sweep_split", "Split sweep input file")
class SweepSplit(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._sweep_split import main

        # Extract arguments
        input = arguments.input

        # Execute sweep split
        main(input)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument("input", type=str, help="Input filepath")
