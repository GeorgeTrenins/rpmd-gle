#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   make_input.py
@Time    :   2024/09/12 14:10:18
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
'''


from __future__ import print_function, division, absolute_import
import argparse
import numpy as np
import json
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
parser.add_argument('--naux', type=int, choices=[1,2,3,4], default=1, help="Number of pairs of auxiliary variables for GLE propagation.")
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
    if args.linear:
        # constant coupling function F(x) = x; L0 scales the spectral density
        coupling_json = {
            "name": "linear",
            "UNITS": u_str
        }
        # convert target zero-frequency value to system units of per-time
        L0 = args.L0 * (u.time / 1.0e-12)
        # undo the mass-weighting
        L0 *= ref_mass
        # data for the mean-field potential
        bath_json = {
            'name'   : 'expohmic',
            'UNITS'  : u_str,
            'Nmodes' : 5000,  
            'eta'    : L0, 
            'omega_cut' :  u.wn2omega(args.wc)
            }
    else:
        # position-dependent coupling, scaling of spectral density is dimensionless 
        friction_data = pd.read_csv(datadir / f'{args.k:02d}_kpts_friction.csv', index_col=None)
        freqs = friction_data.values[:,0]
        Lambda = friction_data.values[:,1:]
        # Broaden the spectral density
        broad_Lambda = broaden_spectral_density(freqs, Lambda, args.s)
        # Extract static friction
        static_friction = broad_Lambda[0]
        xF = np.concatenate([rxn_coord, 2*rxn_coord[-1]-rxn_coord[-2::-1]])
        yF = np.concatenate([static_friction, static_friction[-2::-1]])
        # convert yF from [1/ps] to system units
        yF *= (1.0e12*u.time)
        # FHI-aims returns a mass-weighted EFT - undo the mass weighting
        yF *= ref_mass
        grad_coupling = np.sqrt(yF)
        coupling_json = {
            "name": "splined",
            "x": xF.tolist(),
            "y": grad_coupling.tolist(),
            "mass": mD if args.D else mH,
            "UNITS": "eVAamu",
            "bc_type": "periodic"
        }
        bath_json = {
            'name'   : 'expohmic',
            'UNITS'  : u_str,
            'Nmodes' : 5000,
            'eta'    : args.L0,  # dimensionless
            'omega_cut' :  u.wn2omega(args.wc)
            }
    with open("coupling.json", "w") as f:
        json.dump(coupling_json, f, indent=4)
    with open('bath.json', 'w') as f:
        json.dump(bath_json, f, indent=4)
    
def make_nve(args: argparse.Namespace) -> None:
    from rpmdgle.utils.fitaux import expohmic as parse_coeffs
    import os
    import sys
    fitaux_path = Path(
        f"{os.environ['RPMDGLE']}/auxvars/expohmic/naux_{args.naux:02d}.csv")
    
    # The fitting script measures friction in units of m*wb - we want that to equal 1 in system units (m*wb = 1 => wb = 1/m)
    mass = mD if args.D else mH
    wb = u.omega2wn(1.0/mass)
    if args.linear:
        # convert target zero-frequency value of Lambda to system units of per-time and undo the mass weighting
        eta0 = args.L0 / u.str2base('1 ps') * ref_mass
    else:
        # dimensionless scaling 
        eta0 = args.L0
    fitaux_args = argparse.Namespace(
            data = fitaux_path,
            units = u_str,
            oneaux = (args.naux == 1),
            wc = args.wc,
            mass = mass,
            wb = wb,
            eta0 = eta0,
            T = args.T,
            N = args.N,
            n = None
        )
    nve = "nve.json"
    with open(nve, "w") as sys.stdout:
        print("[\n")
        parse_coeffs.main(fitaux_args)
        print("]")
    sys.stdout = sys.__stdout__
    with open(nve, "r") as f:
        coeffs = json.load(f)
    nve_data = {
        "class": "SepGLEaux",
        "dt": args.dt,
        "aux": coeffs
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
    make_nve(args)
    make_restraint()
    make_nvt()
    make_hist_args()
    make_ti_args()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
