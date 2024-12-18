# Example simulation scripts

The procedure is the same for Ohmic and ab initio (`sigma_20_meV`) spectral densities.

**CAUTION** Unlike in the publication, here, the FCC site is positioned at 0 angstrom and is the "reactant", whereas the HCP site is positioned at 1.5 angstrom and is labelled the "product". 

 1. Change into either subdirectory, depending on which spectral density you wish to use (e.g., `ohmic`).

 1. Run `./make_rpmd_TST.sh` to generate the input files for RPMD TST calculations, which produces a tree of sub-directories with input JSON files and launch scripts based on `ktst.rpmd.template` (modify as needed)

 1. Depending on PES model, isotope, etc., change to the relevant subdirectory  (e.b., `pbe_light_frozen/H/wcut_4000/100K/nbeads_032/eta_10.00/kTST) and execute `bash run.sh &> log.txt &`

 1. Once completed, the final simulation results will appear in `kQTST.json`. The relevant entries are "kQTST", "kQTSTP", "kQTST_err", and "kQTSTP_err", which are the TST rates reactant -> product, product -> reactant, and their respective errors, in units of reciprocal time. The unit system is specified under "UNITS", see `rpmdgle.units' for more details

 1. Similar procedure for transmission coefficients: run `./make_rpmd_cfs.sh` to generate the input files, with launch scripts based on `cfs.rpmd.template` (modify as needed)

 1. For further details on the input file generation, check `python make_input.py -h`

 1. For information on the Bennett-Chandler RPMD simulation scripts, run `python -m rpmdgle.rates.rpmdrate -h`

 1. For analysis of simulation output, see `python -m rpmdgle.rates.analyse -h`
