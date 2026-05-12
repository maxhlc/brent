# Standard imports
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


@ApplicationFactory.register("sem_merge", "Merge SEM files to produce SVN/PRN maps")
class SEMMerge(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # Internal imports
        from ._sem_merge import main

        # Extract arguments
        input = arguments.input
        output = arguments.output

        # Execute SEM merge
        main(input, output)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        # Add arguments to parser
        # TODO: make input list instead of glob?
        parser.add_argument("input", type=str, help="Input filepath")
        parser.add_argument("output", type=str, help="Output filepath")
