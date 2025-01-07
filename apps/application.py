# Standard imports
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

# External imports
from brent.util import Factory


class Application(ABC):

    @staticmethod
    @abstractmethod
    def run(arguments: Namespace) -> None: ...

    @classmethod
    @abstractmethod
    def addArguments(cls, parser: ArgumentParser) -> None: ...

    @classmethod
    def main(cls) -> None:
        # Create parser
        parser = ArgumentParser()
        cls.addArguments(parser)
        arguments = parser.parse_args()

        # Execute application
        cls.run(arguments)


class ApplicationFactory(Factory):

    # TODO: replace with custom factory class?

    @classmethod
    def addArguments(cls, subparsers) -> None:
        # Iterate through applications
        for name, application in cls.registry.items():
            # Add application arguments to subparsers
            subparser = subparsers.add_parser(name)
            application.addArguments(subparser)
