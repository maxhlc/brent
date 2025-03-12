# Future imports
from __future__ import annotations

# Standard imports
from dataclasses import dataclass, asdict


@dataclass
class SEMHeader:

    # Number of records
    records: int

    # File title
    title: str

    # GPS week
    week: int

    # Seconds since start of GPS week
    seconds: int

    def asdict(self) -> dict[str, int | str]:
        # Return header as dictionary
        return asdict(self)

    def serialise(self) -> str:
        # Declare lines
        lines = []

        # Add lines
        lines.append(f"{self.records:02d}  {self.title}")
        #
        lines.append(f" {self.week:04d} {self.seconds:06d}")
        #
        lines.append("")

        # Return serialised header
        return "\n".join(lines)

    @classmethod
    def deserialise(cls, string: str) -> SEMHeader:
        # Split string into lines
        lines = string.splitlines()

        # Deserialise values
        records = int(lines[0][0:2])
        title = lines[0][4:28]
        #
        week = int(lines[1][1:5])
        seconds = int(lines[1][6:12])

        # Return parsed header
        return SEMHeader(records, title, week, seconds)
