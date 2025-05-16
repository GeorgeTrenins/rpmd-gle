# RPMD GLE

Python scripts for simulating RPMD rates in one-dimensional dissipative systems. Please run

```
source env.sh
```

before executing any of the examples.

## Directories

 * `abinitio/pbe_light_frozen/slab+H/06_layers`: energy and friction data from FHI-aims calculations using PBE with the modified `light` species defaults, performed on a six-layer slab along the reaction coordinate with frozen metal substrate
    
    - FHI-aims input files used to compute the energies and electron-phonon coupling (EPC) matrix elements for structures along the reaction coordinate for a 6-layer slab with a 16x16x1 k-grid

    - Corresponding friction spectra (Λ) for every element of the electronic-friction tensor.

    - `convert_epc_to_Lambda.sh` for converting raw EPC files to fricion spectra.

    - `build_pes_and_eft.py` for mapping the energies and EFTs onto the 1D reaction coordinate.

    - potential-energy and friction profiles along the 1D coordinate for k-grids of different sizes

 * `data`: TST rates and transmission coefficients computed using MD and RPMD (see subdirectory README for further details)

 * `figures`: python scripts for generating the figures in the main article

 * `rpmdgle`: scripts for running and processing RPMD rate calculations
 
 * `eftools`: scripts for parsing the EPC elements computed by FHI-aims 

 * `auxvars`: pre-fitted auxiliary variable propagation parameters.

    + `expohmic`: parameters for exponentially damped Ohmic spetral densities

    + `cu111+H`: parameters for ab initio spectral density for Cu(111)+H,
    6-layer slab, 16-point k-grid, 0.02 eV Gaussian broadening

      - `wcut_2000`: density damped by exponential window with 2000 cm<sup>-1</sup> cut-off
      
      - `wcut_4000`: density damped by exponential window with 4000 cm<sup>-1</sup> cut-off

 * `simulations/cu111`: example scripts for setting up RPMD rate calculations. See README in this subdirectory for further details. 
