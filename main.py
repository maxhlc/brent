#!/usr/bin/env python

# Standard imports
from argparse import ArgumentParser

# Internal imports
from apps import ApplicationFactory


def main() -> None:
    # Create parser
    parser = ArgumentParser()

    # Add applications to parser
    subparsers = parser.add_subparsers(
        title="Application",
        dest="application",
        required=True,
    )
    ApplicationFactory.addArguments(subparsers)

    # Parse arguments
    arguments = parser.parse_args()

    # Extract application
    application = ApplicationFactory.create(arguments.application)

    # Execute application
    application.run(arguments)


if __name__ == "__main__":
    # Execute main function
    main()
