# BRENT
BRENT (Batch Regression for Estimating Noise in TLEs) is a tool under development for conducting pseudo-orbit determination on Two-line Element Sets (TLEs).

## Getting Started

### Installation

The following command can be used to clone the repository and its submodules:
```
git clone --recurse-submodules https://github.com/maxhlc/brent
```

Alternatively, the submodules can be initialised at a later point:
```
git submodule update --init --recursive
```

#### Python
An environment file is provided to create a conda environment with all of the required dependencies:

```
conda env create -f environment.yml
```

The conda environment can then be activated:

```
conda activate brent
```

#### Orekit
The Python wrappers for Orekit are distributed through the conda package manager. No further configuration should be required to install Orekit.

Orekit requires data files (including physical parameters, reference frame parameters, etc.) which can be obtained from the [Orekit Data repository](https://gitlab.orekit.org/orekit/orekit-data). These are automatically included as a git submodule.

#### THALASSA
THALASSA must be compiled (including its Python bindings) before it can be used by BRENT. THALASSA uses the CMake build system, and can be configured and built with the following commands:
```
cmake -B ./build -S .
```
```
cmake --build ./build
```

It is recommended to execute these within the conda environment to ensure that the Python bindings are compatible with the environment's version of Python.

By default, BRENT uses SPICE kernels with THALASSA lunisolar perturbations, therefore the following files must be placed into the `./data/kernels/` directory:
* `de431_part-1.bsp`
* `de431_part-2.bsp`
* `gm_de431.tpc`
* `naif0012.tls`

A script for downloading the kernel files is provided by THALASSA (`./external/thalassa/data/kernels/kernels.sh`).

NOTE: SPICE kernels are loaded relative to the current working directory, unlike the Orekit data files which are loaded with absolute filepaths.

### Usage

The main script used with BRENT is `main.py` which is the entry point for various applications in the `./apps/` directory. Many of these use configuration files with examples provided in the `./input/` directory.

TLEs and ILRS/DORIS/IGS orbit product data are expected in JSON and SP3 formats respectively.

## Authors
* Max Hallgarten La Casta (m.hallgarten-la-casta21@imperial.ac.uk)

## License
This project is licensed under the MIT License - see the `LICENSE` file for more details.
