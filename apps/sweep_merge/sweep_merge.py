# Standard imports
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("sweep_merge", "Merge results from multiple sweeps")
class SweepMerge(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._sweep_merge import main

        # Extract arguments
        input = arguments.input
        output = arguments.output

        # Execute sweep merge
        main(input, output)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument(
            "-i",
            "--input",
            action="append",
            type=str,
            required=True,
            help="Input filepath(s)",
        )
        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="Output filepath",
        )
