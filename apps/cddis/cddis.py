# Standard imports
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("cddis", "Download SP3 files from CDDIS")
class CDDIS(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._cddis import main

        # Extract arguments
        input = arguments.input

        # Execute download
        main(input)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument("input", type=str, help="Input filepath")
