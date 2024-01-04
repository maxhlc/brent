# BRENT
BRENT (Batch Regression for Estimating Noise in TLEs) is a tool under development for conducting pseudo-orbit determination on Two-line Element Sets (TLEs).

## Getting Started

### Installation

The Python wrappers for Orekit are distributed through the conda package manager. An environment with the required dependencies can be created from the requirements file in the repository:

```
conda create -n <environment name> --file requirements.txt
```

The conda environment can then be activated:

```
conda activate <environment name>
```

Orekit requires data files (including physical parameters, reference frame parameters, etc.) which can be obtained from the [Orekit Data repository](https://gitlab.orekit.org/orekit/orekit-data). This must be extracted into the `./data/orekit/` directory.

### Usage

Two scripts are provided: `main.py` which conducts a single fit based on command-line arguments, and `sweep.py` which conducts multiple fits based on permutations of supplied arguments defined in the script file.

TLEs and ILRS orbit product data are expected in JSON and SP3 formats respectively.

## Authors
* Max Hallgarten La Casta (m.hallgarten-la-casta21@imperial.ac.uk)

## License
This project is licensed under the MIT License - see the `LICENSE` file for more details.
