#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   build_pes_and_eft.py
@Time    :   2024/08/13 14:59:20
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   Project the EFT onto the reaction coordinate and parse the electronic energies.
'''


from __future__ import print_function, division, absolute_import
import argparse
import numpy as np
from pathlib import Path
from typing import Union


def parse_spectral_density(T: float, datadir: Union[str, Path]) -> np.ndarray:
    Lambda = None
    for i in range(3):
        for j in range(i,3):
            datafile = (Path(datadir) / 
                        f"sigma_0.01_Lambda_atom_000081_cart_{i+1}_000081_cart_{j+1}_temp_{T:06.2f}K.csv")
            try:
                freqs, Lambda_ = np.loadtxt(datafile, skiprows=1, unpack=True, delimiter=",")
            except ValueError:
                print(f"{datafile = }")
                raise
            if Lambda is None:
                Lambda = np.zeros((3, 3, len(Lambda_)))
            Lambda[i, j] = Lambda_
            if i != j:
                Lambda[j, i] = Lambda[i, j]
    return freqs, Lambda


def finite_difference(values, steps, mode="central"):
    if mode == "central":
        if len(values) != 3:
            raise RuntimeError("Must supply the function value at three grid points to perform a central finite difference calculation")
        if len(steps) != 2:
            raise RuntimeError("Must supply two step sizes to perform a central finite difference calculation")
        ans = (values[2] - values[1]) * steps[0]/steps[1]
        ans += (values[1] - values[0]) * steps[1]/steps[0]
        ans /= np.sum(steps, axis=0)
        return ans

    elif mode in {"forward", "backward"}:
        if len(values) != 2:
            raise RuntimeError("Must supply the function value at two grid points to perform a forward/backward finite difference calculation")
        steps = np.atleast_1d(steps)
        if len(steps) != 1:
            raise RuntimeError("Must supply one step size to perform a forward/backward finite difference calculation")
        ans = values[1] - values[0]
        ans /= steps[0]
        return ans
    
    else:
        raise ValueError(f"Unknown mode '{mode}', choose one of 'central', 'forward' or 'backward'")
    
def rxn_coord_and_tangents(adsorbate_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the reaction coordinate, s and tangent grad(s), according to Eqs. (S1)--(S3)."""
    dq_fwd = adsorbate_positions[1:] - adsorbate_positions[:-1]
    ds_fwd = np.sqrt(np.sum(dq_fwd**2, axis=-1))
    s = np.concatenate(([0.0], np.cumsum(ds_fwd)))
    # Use central differences to obtain displacements along reaction coordinate
    # THIS is for projecting the EFT
    tangents = np.empty_like(adsorbate_positions)
    tangents[0] = finite_difference(adsorbate_positions[:2], ds_fwd[:1], mode="forward")
    tangents[0,2] = 0 # z-component
    for i in range(1, len(adsorbate_positions)-1):
        tangents[i] = finite_difference(adsorbate_positions[i-1:i+2], ds_fwd[i-1:i+1], mode="central")
    tangents[-1] = finite_difference(adsorbate_positions[-2:], ds_fwd[-1:], mode="backward")
    tangents[-1,2] = 0 # z-component
    tangent_norm = np.sqrt(np.sum(tangents**2, axis=-1))
    tangents /= tangent_norm[:,None]
    return s, tangents


def main(args: argparse.Namespace) -> None:
    
    import matplotlib.pyplot as plt
    import pandas as pd
    from eftools import parse_energy
    from ase.io import read
    from eftools.units import eVAamu, SI
    plt.style.use('paper.mplstyle')
    import matplotlib as mpl
    mpl.rcParams['lines.linewidth'] = 0.5

    kpts = 16   # number of kpts along one lateral principal direction
    T = 300.0   # electronic temperature
    u: SI = eVAamu()
    wd = Path(__file__).resolve().parent
    rxn_coords: list[Path] = sorted(list(wd.glob('???')))
    adsorbate_positions = []
    total_energies = []
    total_zero_temp = []
    free_energies = []
    spectral_densities = []

    for rxn_coord in rxn_coords:
        cell = read(Path(rxn_coord) / 'geometry.in', format='aims')
        tot, zero, free = parse_energy.main(Path(rxn_coord) / 'aims.out')
        for energy_list, energy_item in zip(
                [total_energies, total_zero_temp, free_energies],
                [tot, zero, free]):
            # Store energies in wavenumbers
            energy_list.append(energy_item[0])
        adsorbate_positions.append([atom.position for atom in cell if atom.symbol == 'H'])
        freqs, Lambda = parse_spectral_density(T, rxn_coord)
        spectral_densities.append(Lambda)
    # Convert excitation energies to wavenumbers
    wn = u.energy2wn(freqs)
    adsorbate_positions = np.asarray(adsorbate_positions).reshape((-1,3))
    s, tangents = rxn_coord_and_tangents(adsorbate_positions)
    projected_Lambda = []
    # Eq. (S4)
    for t, Lambda in zip(tangents, spectral_densities):
        projected_Lambda.append(np.einsum(
            'i,ij...,j->...', t, Lambda, t
        ))
    # Plot the energy 
    total_energies = np.asarray(total_energies)
    total_zero_temp = np.asarray(total_zero_temp)
    free_energies = np.asarray(free_energies)
    fig, (axE, axLambda) = plt.subplots(nrows=2)
    fig.set_size_inches(5, 4)
    for arr, label in zip(
            [total_energies, total_zero_temp, free_energies],
            ['total', r'total (T $\rightarrow$ 0)', 'free']):
        arr -= np.min(arr)
        axE.plot(s, arr*u.energy2wn(1.0), label=label, lw=1)
    axE.legend(loc=1)
    # save the energies
    data = {
        'coordinate [angstrom]' : s.tolist(),
        'total [eV]': total_energies.tolist(),
        'zerotemp [eV]': total_zero_temp.tolist(),
        'free [eV]': free_energies.tolist()
    }
    df = pd.DataFrame(data)
    df.to_csv(f'{kpts:02d}_kpts_energies.csv', index=False)
    vals = np.linspace(0.1, 0.9, len(projected_Lambda))
    colormap_name = 'viridis'
    cmap = plt.get_cmap(colormap_name)
    clist = [cmap(v) for v in vals]
    data = {
        "energies [eV]" : freqs
    }
    for i, (pL, c) in enumerate(zip(projected_Lambda, clist)):
        axLambda.plot(wn, pL, color=c)
        data[f'{s[i]:9.7f}'] = pL
    df = pd.DataFrame(data)
    df.to_csv(f'{kpts:02d}_kpts_friction.csv', index=False)
    axE.set_ylabel('energy [cm$^{-1}$]')
    axE.set_xlabel('reaction coordinate [$\mathrm{\AA}$]')
    axE.set_ylim([-50, 1100])
    axLambda.set_ylabel('$\Lambda(\omega)$ [ps$^{-1}$]')
    axLambda.set_xlabel('$\omega$ [cm$^{-1}$]')
    axLambda.set_ylim([-0.2, 5.2])
    fig.tight_layout()
    plt.show()
    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project the EFT onto the reaction coordinate and parse the electronic "
        "energies. The code parses the `sigma_0.01_*.csv` files from the directories `???` and "
        "produces a 4-column file 16_kpts_energies.csv (see header for explanation) and " 
        "a multicolumn file 16_kpts_friction.csv with the first column specifying the excitation energy in eV and the susbequent columns giving the friction spectrum Λ divided by the "
        "mass of the hydrogen atom mH, in [1/ps]."
    )
    args = parser.parse_args()
    main(args)
