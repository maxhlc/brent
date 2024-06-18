# Future imports
from __future__ import annotations

# Standard imports
from datetime import datetime
import json
from typing import Any, Dict

# Third-party imports
import pandas as pd

# Internal imports
from brent.propagators import Propagator, deserialise_propagator


class Fit:

    def __init__(
        self,
        dates: FitDates,
        observer: Propagator,
        validator: Propagator,
        filter: FitFilter,
    ):
        # Store dates
        self.dates = dates

        # Store propagators
        self.observer = observer
        self.validator = validator

        # Store filter
        self.filter = filter

    def serialise(self):
        # Return serialised object
        return {
            "dates": self.dates.serialise(),
            "propagators": {
                "observer": self.observer.serialise(),
                "validator": self.validator.serialise(),
            },
            "filter": self.filter.serialise(),
        }

    @staticmethod
    def deserialise(struct):
        # Deserialise dates
        dates = FitDates.deserialise(struct["dates"])

        # Deserialise propagators
        observer = deserialise_propagator(struct["propagators"]["observer"])
        validator = deserialise_propagator(struct["propagators"]["validator"])

        # Deserialise filter
        filter = FitFilter.deserialise(struct["filter"])

        # Return deserialised
        return Fit(dates, observer, validator, filter)

    @staticmethod
    def load(fname: str) -> Fit:
        # Load serialised object
        with open(fname, "r") as fp:
            struct = json.load(fp)

        # Return deserialised object
        return Fit.deserialise(struct)

    def save(self, fname: str):
        # Serialise object
        struct = self.serialise()

        # Save to file
        with open(fname, "w") as fp:
            json.dump(struct, fp, indent=4)


class FitDates:
    def __init__(self, start: datetime, end: datetime, nsamples: int) -> None:
        # Assert validity of inputs
        assert start < end
        assert nsamples > 0

        # Store parameters
        self.start = start
        self.end = end
        self.nsamples = nsamples

        # Generate dates
        self.dates = pd.date_range(start, end, nsamples)

    def serialise(self) -> Dict[str, int | str]:
        # Return serialised form
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "nsamples": self.nsamples,
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> FitDates:
        # Extract parameters
        start = datetime.fromisoformat(struct["start"])
        end = datetime.fromisoformat(struct["end"])
        nsamples = int(struct["nsamples"])

        # Return fit dates
        return FitDates(start, end, nsamples)


class FitFilter:

    def serialise(self):
        pass

    @staticmethod
    def deserialise(struct) -> FitFilter:
        return FitFilter()
