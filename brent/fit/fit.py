# Future imports
from __future__ import annotations

# Standard imports
from datetime import datetime
import json
from typing import Dict, Any, Type

# Third-party imports
import pandas as pd

# Internal imports
from brent import Constants
from brent.bias import BiasModel, get_bias_type
from brent.noise import CovarianceProvider, get_noise_type
from brent.propagators import Propagator, WrappedPropagator, get_propagator_type


class Fit:

    def __init__(
        self,
        dates: FitDates,
        observer: FitPropagator,
        fit: FitPropagatorBuilder,
        validator: FitPropagator,
        filter: FitFilter,
    ):
        # Store dates
        self.dates = dates

        # Store propagators
        self.observer = observer
        self.fit = fit
        self.validator = validator

        # Store filter
        self.filter = filter

    def serialise(self):
        # Return serialised object
        return {
            "dates": self.dates.serialise(),
            "propagators": {
                "observer": self.observer.serialise(),
                "fit": self.fit.serialise(),
                "validator": self.validator.serialise(),
            },
            "filter": self.filter.serialise(),
        }

    @staticmethod
    def deserialise(struct):
        # Deserialise dates
        dates = FitDates.deserialise(struct["dates"])

        # Deserialise propagators
        observer = FitPropagator.deserialise(struct["propagators"]["observer"])
        fit = FitPropagatorBuilder.deserialise(struct["propagators"]["fit"])
        validator = FitPropagator.deserialise(struct["propagators"]["validator"])

        # Deserialise filter
        filter = FitFilter.deserialise(struct["filter"])

        # Return deserialised
        return Fit(dates, observer, fit, validator, filter)

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
            json.dump(struct, fp)


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


class FitPropagator(WrappedPropagator):

    def __init__(
        self,
        propagator: Propagator,
        noise: CovarianceProvider,
        bias: BiasModel,
    ):
        # Store parameters
        self.propagator = propagator
        self.noise = noise
        self.bias = bias

        # Initialise underlying propagator
        super().__init__(propagator)

    def propagate(self, dates, frame=Constants.DEFAULT_ECI):
        # Propagate biased state
        biased = self.propagator.propagate(dates, frame)

        # Calculate bias correction
        if self.bias == None:
            states = biased
        else:
            states = self.bias.debias(dates, biased)

        # Return states
        return states

    def serialise(self) -> Dict[str, Any]:
        # TODO: add results
        return {
            "propagator": self.propagator.serialise(),
            "noise": self.noise.serialise(),
            "bias": self.bias.serialise(),
        }

    @staticmethod
    def deserialise(struct: Dict[str, Any]) -> FitPropagator:
        # Extract structs
        propagator_ = struct.get("propagator", None)
        noise_ = struct.get("noise", None)
        bias_ = struct.get("bias", None)

        # Build noise model
        noiseType = get_noise_type(noise_["type"])
        noise = noiseType.deserialise(noise_)

        # Build bias model
        biasType = get_bias_type(bias_["type"])
        bias = biasType.deserialise(bias_)

        # Build propagator model
        propagatorType = get_propagator_type(propagator_["type"])
        propagator = propagatorType.deserialise(propagator_)

        # Return overall model
        return FitPropagator(propagator, noise, bias)


class FitPropagatorBuilder:

    def serialise(self):
        pass

    @staticmethod
    def deserialise(struct) -> FitPropagatorBuilder:
        return FitPropagatorBuilder()


class FitFilter:

    def serialise(self):
        pass

    @staticmethod
    def deserialise(struct) -> FitFilter:
        return FitFilter()
