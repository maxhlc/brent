# Standard imports
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("sem_download", "Download SEM files from Celestrak")
class SEMDownload(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._sem_download import main

        # Extract arguments
        output = arguments.output

        # Execute SEM file download
        main(output)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        parser.add_argument("output", type=str, help="Output filepath")
