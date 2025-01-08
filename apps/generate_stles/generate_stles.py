# Standard imports
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("generate_stles", "Generate S-TLEs from SP3 data")
class GenerateSTLEs(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._generate_stles import main

        # Extract arguments
        input = arguments.input

        # Execute S-TLE generation
        main(input)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument("input", type=str, help="Input filepath")
