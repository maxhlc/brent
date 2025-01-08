# Standard imports
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("sweep_plot", "Plot sweep results")
class SweepPlot(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._sweep_plot import main

        # Extract arguments
        input = arguments.input

        # Execute sweep plot
        main(input)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument("input", type=str, help="Input filepath")
