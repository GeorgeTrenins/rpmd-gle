# RPMD GLE

Python scripts for simulating RPMD rates in one-dimensional dissipative systems. Please run

```
source env.sh
```

before executing any of the examples.

## Directories

 * `abinitio`: potential energy and friction obtained from DFT calculations with FHI-aims.

 * `rpmdgle`: scripts for running and processing RPMD rate calculations

 * `auxvars`: pre-fitted auxiliary variable propagation parameters.

    + `expohmic`: parameters for exponentially damped Ohmic spetral densities

    + `cu111+H/wcut_2000` and `cu111+H/wcut_4000`: parameters for ab initio spectral density for Cu(111)+H,
    6-layer slab, 16-point k-grid, 0.02 eV Gaussian broadening

 * `simulations/cu111`: example scripts for setting up RPMD rate calculations. See README in this subdirectory for further details. 
