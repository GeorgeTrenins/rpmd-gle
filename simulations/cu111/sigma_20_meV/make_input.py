#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   make_input.py
@Time    :   2024/09/12 14:10:18
@Author  :   George Trenins
'''


from __future__ import print_function, division, absolute_import
import argparse
import numpy as np
import json
import pandas as pd
from rpmdgle import units
from rpmdgle.constants import mH, mD
from pathlib import Path


parser = argparse.ArgumentParser(description="Prepare the input files for rate calculations", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('model', help="Specify a directory from `rpmd-gle/abinitio`")
parser.add_argument('bias', type=float, help="Positive energy bias, in eV, applied at the edges of the potential energy curve.")
parser.add_argument('-l', type=int, default=6, help="Number of metal layers")
parser.add_argument('-k', type=int, default=16, help="Size of k-grid.")
parser.add_argument('-s', type=float, default=0.3, help="Target value of `friction_broadening_width`, in eV.")
parser.add_argument('--wc', type=float, default=2000, help="Cut-off frequency for the exponential damping envelope of the spectral density.")
parser.add_argument('--naux', type=int, default=1, help="Number of pairs of auxiliary variables for GLE propagation.")
parser.add_argument('-T', type=float, default=300.0, help="Temperature in Kelvin")
parser.add_argument('-N', type=int, default=1, help='Number or ring-polymer beads')
parser.add_argument('--dt', type=str, default="0.50 fs", help='integration time-step')
parser.add_argument('--tau', type=str, default="10 fs", help='centroid thermostatting constant')
parser.add_argument('--L0', type=float, default=1.0, help='When accounting for the spatial variation of the friction, then the scaling of the friction tensor (dimensionless); otherwise the value of static friction for hydrogen-1 in 1/ps.')
parser.add_argument('--linear', action="store_true", help='Ignore spatial variation of friction, use linear coupling')
parser.add_argument('--D', action="store_true", help='calculate the rate for a deuterium atom instead of protium')
parser.add_argument('--show', action="store_true", help='Show plot windows')

# "Reference" mass (of H-1) in amu, for parsing electronic friction data
ref_mass = mH  
u_str = "eVAamu"
u: units.SI = getattr(units, u_str)()
# Dividing surface and reactant/product minima - determined later in the script
xdd = 0.0 # dividing surface
xR = 0.0 # reactant
xP = 0.0 # product
wd = Path(__file__).resolve().parent

def broaden_spectral_density(
        freqs: np.ndarray, 
        Lambda: np.ndarray, 
        sigma: float) -> np.ndarray: 
    from eftools.friction import FrictionParser
    assert (freqs.ndim == 1) and (Lambda.ndim == 2)
    ans = np.zeros_like(Lambda)
    width = 0.01 # friction_window_size in eV pre-broadening
    for L1, L2 in zip(Lambda.T, ans.T):
        L2[:] = FrictionParser.broaden_friction_tensor_grid(
            None, sigma, freqs, L1, width)[1]
    return ans

def make_pes(args: argparse.Namespace) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from scipy.interpolate import CubicSpline

    datadir = ( Path(os.environ["RPMDGLE"]) / "abinitio" / 
               f"{args.model}" / "slab+H" / f"{args.l:02d}_layers" )
    pes_topup = args.bias

    #
    # POTENTIAL ENERGY
    #

    # Load energy and friction data
    energy_data = pd.read_csv(
        datadir / f"{args.k:02d}_kpts_energies.csv", index_col=None)
    # Build the double well model
    rxn_coord = energy_data["coordinate [angstrom]"].to_numpy()
    energy =  energy_data["free [eV]"].to_numpy()
    # tile
    slc = slice(6,-6,None)
    xE = np.concatenate([-rxn_coord[-1:0:-1], rxn_coord, 2*rxn_coord[-1]-rxn_coord[-2::-1]])[slc]
    yE = np.concatenate([energy[-1:0:-1], energy, energy[-2::-1]])[slc].copy()
    yE[0] += pes_topup
    yE[-1] += pes_topup
    energy_cs = CubicSpline(xE, yE, bc_type="not-a-knot", extrapolate=True)
    grad_cs = energy_cs.derivative(nu=1)
    global xR, xdd, xP
    try: 
        xR, xdd, xP = grad_cs.roots()
    except ValueError:
        raise ValueError("The PES has the wrong number of stationary points (expecting 3). Try tweaking the bias")
    print(f"{xR = }, {xdd = }, {xP = }")
    
    energy_json = {
        "module": "rpmdgle.pes.splined",
        "name": "OneDCubic",
        "x": xE.tolist(),
        "y": yE.tolist(),
        "mass": mD if args.D else mH,
        "UNITS": u_str,
        "bc_type": "not-a-knot", 
        "extrapolate": True
    }
    with open("potential.json", "w") as f:
        json.dump(energy_json, f, indent=4)

    #
    # ELECTRONIC FRICTION
    #

    coupling_json = {
        "name": "linear",
        "UNITS": u_str
    }
    with open("coupling.json", "w") as f:
        json.dump(coupling_json, f, indent=4)

    friction_data = pd.read_csv(datadir / f'{args.k:02d}_kpts_friction.csv', index_col=None)
    freqs = friction_data.values[:,0]
    # Broaden the spectral density
    Lambda_data = broaden_spectral_density(freqs, friction_data.values[:,1:], args.s)
    # scale by the value at zero frequency
    Lambda_data /= Lambda_data[0,None,:]
    # convert energy to frequency
    freqs /= u.hbar 
    # construct a cubic spline interpolation
    mirrored_freqs = np.concatenate([-freqs[:0:-1], freqs])
    idx = 0 # use spectral density from fcc site
    mirrored_Lambda_data = np.concatenate([Lambda_data[:0:-1,idx], Lambda_data[:,idx]])
    # apply window
    wc = u.wn2omega(args.wc)
    # convert target zero-frequency value to system units of per-time
    L0 = args.L0 * (u.time / 1.0e-12)
    # undo the mass-weighting
    L0 *= ref_mass
    mirrored_Lambda_data *= np.exp(-np.abs(mirrored_freqs/wc))
    density_json = {
        "name": "splined",
        "Nmodes": 5000,
        "x": mirrored_freqs.tolist(),
        "y": mirrored_Lambda_data.tolist(),
        "eta": L0,
        "eps": 1.0e-8,
        "UNITS": u_str
    }
    with open("bath.json", "w") as f:
        json.dump(density_json, f, indent=4)

def get_wn_params(wn: float, tol: float, df: pd.DataFrame, nO: int) -> tuple[list, bool]:
    try:
        idx = np.argmin(np.abs(df[('nmfreq', 0)] - wn))
    except ValueError:
        is_close = False
        row = None
    else:
        row = df.loc[idx]
        matched_freq = row[('nmfreq', 0)]
        if abs(matched_freq - wn) < tol:
            is_close = True
        else:
            is_close = False
    return row, is_close

    
def make_nve(args: argparse.Namespace) -> None:
    from rpmdgle.utils.fitaux import expohmic as parse_coeffs
    import os
    import sys
    fitaux_path = Path(
        f"{os.environ['RPMDGLE']}/auxvars/cu111+H/wcut_{int(args.wc)}/naux_{args.naux:02d}.csv")
    if fitaux_path.is_file():
        df_ = pd.read_csv(
            fitaux_path, 
            delimiter=',',
            header=None, 
            index_col=None)
    else:
        raise RuntimeError(f"Could not find the requested auxvar parameter file '{str(fitaux_path)}'")
    new_labels = pd.MultiIndex.from_arrays(
        [df_.iloc[0], np.asarray(df_.iloc[1].astype(float), dtype=int)],
        names=['parameter', 'index'])
    # reorganize the dataframe so that it has columns with multiindex labels
    # (nmfreq, 0), (tauO, 0), (tauO, 1), ..., (cO, 0), (cO, 1), ..., (omegaO, 0), (omegaO, 1), ...
    df = df_.set_axis(new_labels, axis=1).iloc[2:].astype(float)    
    df.reset_index(inplace=True)
    del df['index']
    # cycle over the normal-mode frequencies
    beta = u.betaTemp(args.T)
    wn_lst = parse_coeffs.get_nm_freqs(beta, u.hbar, args.N)
    tol = 1.0e-4/(beta*u.hbar) # tolerance for frequency matching is a small fraction of the first Matsubara frequency
    data = []
    # The fit is to a spectral density scaled such that the zero-frequency value is 1
    # L0 is the rescaled target value.
    L0 = args.L0 * (u.time / 1.0e-12) * ref_mass
    for n, wn in enumerate(wn_lst):
        df_row, is_close = get_wn_params(wn, tol, df, args.naux)
        if not is_close:
            raise RuntimeError(f"Could not find all pre-fitted parameters for N = {args.N} at T = {args.T} K.")
        df_row = df_row.to_numpy()
        data.append(
            {
                "tauO" : df_row[1:args.naux+1].tolist(),
                "cO" :  (df_row[args.naux+1:2*args.naux+1] * np.sqrt(L0)).tolist(),
                "omegaO" : df_row[2*args.naux+1:].tolist()
            }
        )
    nve = "nve.json"
    nve_data = {
        "class": "SepGLEaux",
        "dt": args.dt,
        "aux": data
    }
    with open(nve, "w") as f:
        json.dump(nve_data, f, indent=4)

def make_dummy_nve(args: argparse.Namespace) -> None:
    nve = "nve.json"
    nve_data = {
        "class": 'SepGLEPILE',
        'dt'    : args.dt
    }
    with open(nve, "w") as f:
        json.dump(nve_data, f, indent=4)
    

def make_restraint():
    from rpmdgle.rates.qtst import get_PES, RestrainedRing
    import matplotlib.pyplot as plt

    with open("potential.json", 'r') as f:
        params = json.load(f)

    PES, UNITS = get_PES(params)
    x = np.linspace(-1.5,3,501)
    shift = 0.005
    k = 50.0
    rpPES = RestrainedRing(xdd - shift, k, -1, N=1, xshape=(1,), extPES=PES, beta=u.betaTemp(300.0))
    if args.show:
        plt.plot(x, PES.potential(x[:,None]))
        plt.plot(x, rpPES.potential(x[:,None,None]))
        plt.show()
    specs = {
        'shift' : xdd - shift,
        'k'     : k,
        'direction' : -1
    }
    with open('restraint_R.json', 'w') as f:
        json.dump(specs, f, indent=4)
    
    specs['shift'] = xdd + shift
    specs['direction'] = 1
    with open('restraint_P.json', 'w') as f:
        json.dump(specs, f, indent=4)

def make_nvt():
    specs = {
        'class' : 'SepGLEPILE',
        'dt'    : args.dt,
        'tau'   : args.tau
        }
    with open('nvt.json', 'w') as f:
        json.dump(specs, f, indent=4)

def make_hist_args():
    propname = 'hist_properties.json'
    propspecs = [{
        "A" : 'xnm0',
        "postA" : "lambda arr: arr[...,0]",
        "stride" : '25 fs',
        "name" : "coords"
    }]
    with open(propname, 'w') as f:
        json.dump(propspecs, f, indent=4)

def make_ti_args():
    propname = 'ti_properties.json'
    propspecs = [{
        "A" : 'fnm0',
        "postA" : "lambda arr: arr[...,0]",
        "stride" : '25 fs',
        "name" : "forces",
    }]
    with open(propname, 'w') as f:
        json.dump(propspecs, f, indent=4)

def main(args: argparse.Namespace) -> None:
    make_pes(args)
    if args.naux > 0:
        make_nve(args)
    else:
        make_dummy_nve(args)
    make_restraint()
    make_nvt()
    make_hist_args()
    make_ti_args()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
