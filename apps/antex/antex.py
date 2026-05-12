# Standard imports
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("antex", "Parse ANTEX file to produce COSPAR/PRN maps")
class Antex(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._antex import main

        # Extract arguments
        input = arguments.input
        output = arguments.output

        # Execute SEM merge
        main(input, output)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument("input", type=str, help="Input filepath")
        parser.add_argument("output", type=str, help="Output filepath")
