# Standard imports
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace


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


class ApplicationFactory:

    # Registry of classes
    registry = {}

    @classmethod
    def register(cls, name: str, help: str):
        def inner_wrapper(class_type):
            # Raise error if class name already registered
            if name in cls.registry:
                raise RuntimeError(f"Class already registered: {name}")

            # Store class name and type
            cls.registry[name] = {
                "class": class_type,
                "help": help,
            }

            # Return class
            return class_type

        # Return inner function
        return inner_wrapper

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        # Raise error if not in registry
        if name not in cls.registry:
            raise RuntimeError(f"Unknown class: {name}")

        # Extract class
        class_type = cls.registry[name]["class"]

        # Return initialised class
        return class_type(*args, **kwargs)

    @classmethod
    def addArguments(cls, subparsers) -> None:
        # Iterate through applications
        for name, application in cls.registry.items():
            # Add application arguments to subparsers
            subparser = subparsers.add_parser(name, help=application["help"])
            application["class"].addArguments(subparser)
