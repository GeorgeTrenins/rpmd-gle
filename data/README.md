# Cu(111)+H escape rates

 * `townsend_cu111_H_rates.csv`: hcp well escape rates for H from HeSE experiments; data from Fig. 6.40 in Townsend, P. S. M. (2018). *Diffusion of light adsorbates on transition metal surfaces.* [https://doi.org/10.17863/CAM.21077](https://doi.org/10.17863/CAM.21077)

 * `townsend_cu111_D_rates.csv`: same for D

 * `ohmic`, `sigma_20_meV` and `sigma_40_meV` contain results of RPMD rate simulations using Ohmic friction, ab initio friction with 0.02 eV Gaussian spectral broadening, and 0.04 eV broadening, respectively

   + subdirectories `H` / `D` contain results for hydrogen and deuterium, respectively

   + `wcut_2000` / `wcut_4000` contain results for spectral densities damped by an exponential window with a cut-off of 2000 and 4000 cm^-1^, respectively

   + the data are then stored in a directory tree, sorted by temperature -> number of ring-polymer beads -> scaling of ab initio frequency

   + QTST simulation results are stored as JSON files; `kQTST` is the TST rate for *fcc* -> *hcp*, `kQTSTP` is the TST rate for *hcp* -> *fcc*. See `rpmdgle.rates.analyse` for further details

   + Flux-side time correlation functions are stored as CSV files; the first column is time (in ps); the next two columns are the numerator in the definition of the transmission coefficient, e.g., Eq. (47) of the paper by [Collepardo-Guevara et al.](https://doi.org/10.1063/1.2883593) and its sampling error estimate. The denominator and its error are given in the comment line; the last two columns are the transmission coefficient and its error

