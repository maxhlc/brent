# Future imports
from __future__ import annotations

# Standard imports
from datetime import datetime
from functools import cache

# Third-party imports
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from org.orekit.data import DataSource
from org.orekit.gnss import SatelliteSystem
from org.orekit.gnss.antenna import AntexLoader, SatelliteAntenna
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import TimeSpanMap


class AntexFile:

    def __init__(self, antex: AntexLoader) -> None:
        # Store ANTEX file
        self.antex = antex

    @classmethod
    def load(cls, name: str) -> AntexFile:
        # Load ANTEX file
        datasource = DataSource(name)
        timescale = TimeScalesFactory.getGPS()
        antex = AntexLoader(datasource, timescale)

        # Return ANTEX object
        return AntexFile(antex)

    @cache
    def get_prn(self) -> pd.DataFrame:
        # Get antennas
        # NOTE: extracting the antenna spans requires brute forcing by date, due to
        #       the lack of a method which provides all spans in one call
        dates = pd.date_range(
            start=datetime(1975, 1, 1),
            end=self.NOW,
            freq="1D",  # TODO: replace with different frequency?
        )
        dates.append(pd.DatetimeIndex([self.NOW]))

        # Extract antenna spans
        antennaSpans = list(self.antex.getSatellitesAntennas())
        antennaSpans = [TimeSpanMap.cast_(antennaSpan) for antennaSpan in antennaSpans]

        # Get antennas
        antennas = [
            SatelliteAntenna.cast_(antennaSpan.get(datetime_to_absolutedate(date)))
            for date in dates
            for antennaSpan in antennaSpans
        ]

        # Get unique antennas
        antennas = set(antennas)

        # Create DataFrame
        records = [
            {
                # Fixed properties
                "cospar": antenna.getCosparID(),
                "system": antenna.getSatelliteSystem(),
                # Assignable properties
                "prn": antenna.getPrnNumber(),
                # Dates
                "start": self._date(antenna.getValidFrom()),
                "end": self._date(antenna.getValidUntil()),
            }
            for antenna in antennas
        ]
        df = pd.DataFrame(records)

        # Generate SP3 identifiers
        df["system"] = df["system"].map(self.GNSS_SYSTEM_MAP)
        df["sp3"] = df.apply(lambda x: f"{x.system}{x.prn:02d}", axis=1)

        # Sort rows
        df.sort_values(["system", "cospar", "start"], inplace=True, ignore_index=True)

        # Sort columns
        df = df[["cospar", "start", "end", "system", "prn", "sp3"]]

        # Return
        return df

    @classmethod
    def _date(cls, date: AbsoluteDate) -> datetime:
        try:
            # Return casted datetime object
            return absolutedate_to_datetime(date)
        except:
            # Return current datetime
            # NOTE: needed as one-sided timespans in the ANTEX file are interpreted
            #       with an end date which cannot be represented by datetime
            return cls.NOW

    # Current datetime
    NOW = datetime.now()

    # Map from GNSS systems to SP3 prefixes
    # TODO: confirm entries
    GNSS_SYSTEM_MAP = {
        SatelliteSystem.BEIDOU: "C",
        SatelliteSystem.GALILEO: "E",
        SatelliteSystem.GLONASS: "R",
        SatelliteSystem.GPS: "G",
        SatelliteSystem.IRNSS: "I",
        SatelliteSystem.MIXED: "M",
        SatelliteSystem.QZSS: "J",
        SatelliteSystem.SBAS: "",
    }
