# Future imports
from __future__ import annotations

# Third-party imports
import pandas as pd

# Internal imports
from .header import SEMHeader
from .record import SEMRecord


class SEMFile:

    def __init__(self, header: SEMHeader, records: list[SEMRecord]):
        # Check for correct number of records
        if header.records != len(records):
            raise ValueError("Number of records does not match header")

        # Store header and records
        self.header = header
        self.records = records

    def to_dataframe(self) -> pd.DataFrame:
        # Convert header and records to dictionaries
        header = self.header.asdict()
        records = [record.asdict() for record in self.records]

        # Return file as DataFrame
        return pd.DataFrame([{**header, **record} for record in records])

    def serialise(self) -> str:
        # Declare file sections
        sections = []

        # Serialise file sections
        sections.append(self.header.serialise())
        sections.extend([record.serialise() for record in self.records])

        # Return joined sections
        return "\n".join(sections)

    @classmethod
    def deserialise(cls, string: str, year: None | int = None) -> SEMFile:
        # Split string into sections
        sections = string.split("\n\n")

        # Remove any blank sections
        sections = [section for section in sections if section != ""]

        # Deserialise header
        header = SEMHeader.deserialise(sections[0], year)

        # Deserialise records
        records = [SEMRecord.deserialise(section) for section in sections[1:]]

        # Return deserialised file
        return SEMFile(header, records)

    def save(self, name: str) -> None:
        # Serialise file
        string = self.serialise()

        # Open file
        with open(name, "w") as fp:
            # Save serialised file
            fp.write(string)

    @classmethod
    def load(cls, name: str, year: None | int = None) -> SEMFile:
        # Open file
        with open(name) as fp:
            # Read contents
            string = fp.read()

        # Return deserialised file
        return SEMFile.deserialise(string, year)
