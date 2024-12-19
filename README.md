# RPMD GLE

Python scripts for simulating RPMD rates in one-dimensional dissipative systems. Please run

```
source env.sh
```

before executing any of the examples.

## Directories

 * `abinitio`: potential energy and friction obtained from DFT calculations with FHI-aims.

 * `data`: TST rates and transmission coefficients computed using MD and RPMD (see subdirectory README for further details)

 * `figures`: python scripts for generating the figures in the main article

 * `rpmdgle`: scripts for running and processing RPMD rate calculations

 * `auxvars`: pre-fitted auxiliary variable propagation parameters.

    + `expohmic`: parameters for exponentially damped Ohmic spetral densities

    + `cu111+H`: parameters for ab initio spectral density for Cu(111)+H,
    6-layer slab, 16-point k-grid, 0.02 eV Gaussian broadening

      - `wcut_2000`: density damped by exponential window with 2000 cm<sup>-1</sup> cut-off
      
      - `wcut_4000`: density damped by exponential window with 4000 cm<sup>-1</sup> cut-off

 * `simulations/cu111`: example scripts for setting up RPMD rate calculations. See README in this subdirectory for further details. 
