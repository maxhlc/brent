# Future imports
from __future__ import annotations

# Standard imports
from dataclasses import dataclass, asdict


@dataclass
class SEMRecord:

    # PRN number
    prn: int

    # Satellite reference number
    svn: int

    # Average URA number
    ura: int

    # Eccentricty
    ecc: float

    # Inclination offset
    inc: float

    # Rate of right ascension
    omd: float

    # Square root of the semi-major axis
    sma: float

    # Longitude of orbital plane
    lon: float

    # Argument of perigee
    aop: float

    # Mean anomaly
    ma0: float

    # Zeroth-order clock correction
    af0: float

    # First-order clock correction
    af1: float

    # Satellite health
    health: int

    # Satellite configuration
    config: int

    def asdict(self) -> dict[str, int | float]:
        # Return record as dictionary
        return asdict(self)

    def serialise(self) -> str:
        # Declare lines
        lines = []

        # Add lines
        lines.append(f"{self.prn}")
        #
        lines.append(f"{self.svn}")
        #
        lines.append(f"{self.ura}")
        #
        lines.append(f"{self.ecc: 1.14E} {self.inc: 1.14E} {self.omd: 1.14E}")
        #
        lines.append(f"{self.sma: 1.14E} {self.lon: 1.14E} {self.aop: 1.14E}")
        #
        lines.append(f"{self.ma0: 1.14E} {self.af0: 1.14E} {self.af1: 1.14E}")
        #
        lines.append(f"{self.health}")
        #
        lines.append(f"{self.config}")
        #
        lines.append("")

        # Return serialised record
        return "\n".join(lines)

    @classmethod
    def deserialise(cls, string: str) -> SEMRecord:
        # Split string into lines
        lines = string.splitlines()

        # Deserialise values
        prn = int(lines[0][0:2])
        #
        svn = int(lines[1][0:3])
        #
        ura = int(lines[2][0:2])
        #
        ecc = float(lines[3][0:21])
        inc = float(lines[3][22:43])
        omd = float(lines[3][44:66])
        #
        sma = float(lines[4][0:21])
        lon = float(lines[4][22:43])
        aop = float(lines[4][44:66])
        #
        ma0 = float(lines[5][0:21])
        af0 = float(lines[5][22:43])
        af1 = float(lines[5][44:66])
        #
        health = int(lines[6][0:2])
        config = int(lines[7][0:2])

        # Return deserialised record
        return SEMRecord(
            prn,
            svn,
            ura,
            ecc,
            inc,
            omd,
            sma,
            lon,
            aop,
            ma0,
            af0,
            af1,
            health,
            config,
        )
