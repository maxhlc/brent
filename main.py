# Internal imports
import brent


def main():
    # Load TLEs
    tles = brent.io.load_tle("./data/tles/8820.json")

    # Load SLR data
    slr = brent.io.load_sp3("./data/sp3/lageos1/*.lageos1.*.sp3", "L51")


if __name__ == "__main__":
    main()
