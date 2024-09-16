# BRENT
BRENT (Batch Regression for Estimating Noise in TLEs) is a tool under development for conducting pseudo-orbit determination on Two-line Element Sets (TLEs).

## Getting Started

### Installation

#### Python
An environment with the required dependencies can be created from the requirements file in the repository:

```
conda create -n <environment name> --file requirements.txt
```

The conda environment can then be activated:

```
conda activate <environment name>
```

#### Orekit
The Python wrappers for Orekit are distributed through the conda package manager. No further configuration should be required to install Orekit.

Orekit requires data files (including physical parameters, reference frame parameters, etc.) which can be obtained from the [Orekit Data repository](https://gitlab.orekit.org/orekit/orekit-data). This must be extracted into the `./data/orekit/` directory.

#### THALASSA
THALASSA must be compiled (including its Python bindings) before it can be used by BRENT. THALASSA uses the CMake build system, and can be configured and built with the following commands:
```
cmake -B ./external/thalassa/build -S ./external/thalassa
```
```
cmake --build ./external/thalassa/build
```

It is recommended to execute these within the conda environment to ensure that the Python bindings are compatible with the environment's version of Python.

By default, BRENT uses THALASSA's simple ephemerides for lunisolar perturbations, therefore additional data files (e.g., SPICE kernels) do not need to be downloaded.

### Usage

The main script used with BRENT is `sweep.py` which conducts multiple fits based on permutations of supplied arguments. An example of a configuration file is available at `./input/sweep.json.example`.

TLEs and ILRS orbit product data are expected in JSON and SP3 formats respectively.

## Authors
* Max Hallgarten La Casta (m.hallgarten-la-casta21@imperial.ac.uk)

## License
This project is licensed under the MIT License - see the `LICENSE` file for more details.
