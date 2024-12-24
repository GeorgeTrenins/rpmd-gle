#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   figure01.py
@Time    :   2024/10/22 10:49:18
@Author  :   George Trenins
@Desc    :   Plot of Cu(111) + diffusing hydrogen atom,  model potential, spectral densities, and QTST rates showing the difference between Ohmic and two non-Markovian spectral density models
'''


from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')
import matplotlib.image as mpimg
from pathlib import Path
from rpmdgle.pes.splined import OneDCubic as Splined
from rpmdgle.units import SI
from rpmdgle.utils.aims.friction import FrictionParser
import json
import numpy as np
import mycolours
import pandas as pd
import glob
import string
import os

data = Path(os.environ['RPMDGLE']) / 'data'
aims = Path(os.environ['RPMDGLE']) / 'abinitio' / 'pbe_light_frozen' / 'slab+H' / '06_layers'

def get_kQTST_data(model, temp, N):
    path = data / model / f"{temp:03d}K" / f"nbeads_{N:03d}"
    etas = glob.glob(str(path / "eta_??.??"))
    etas.sort()
    eta_list = []
    kQTST_list = []
    for eta_dir in etas:
        eta_dir = Path(eta_dir)
        eta = float(eta_dir.name[-5:])
        eta_list.append(eta)
        with open(eta_dir / "kQTST.json", 'r') as f:
            d = json.load(f)
            kQTST_list.append(d["kQTSTP"])
    return np.asarray(eta_list), np.asarray(kQTST_list)

fig = plt.figure(figsize=(3.375, 4.6))

with open(data / 'ohmic' / 'pbe_light_frozen' / 'H' / 'potential.json', 'r') as f:
    pot = json.load(f)
pot.pop("module")
pot.pop("name")
PES = Splined(**pot)
ax0 = plt.subplot2grid((7, 6), (0, 0), colspan=3, rowspan=2)
ax1 = plt.subplot2grid((7, 6), (0, 3), colspan=3, rowspan=2)
ax2 = plt.subplot2grid((7, 6), (2, 0), colspan=3, rowspan=2)
ax3 = plt.subplot2grid((7, 6), (2, 3), colspan=3, rowspan=2, sharex=ax2, sharey=ax2)
ax4 = plt.subplot2grid((7, 6), (4, 0), colspan=6, rowspan=3)

axes = [ax1, ax0, ax2, ax3, ax4]

xgrid = np.linspace(-0.85, 2.35, 501)
u: SI = PES.UNITS
pot = PES.potential(xgrid[:,None]) * 1000
# make hcp site the reactant -- reverse order
pot = pot[::-1]
xgrid = -xgrid[::-1]
# set well at origin
mask = np.abs(xgrid + 1.5) < 0.25
minR = np.min(pot[mask])
idxR = np.argmin(np.abs(pot-minR))
xR = xgrid[idxR]
xgrid -= xR
ax0.plot(xgrid, pot, c=mycolours.silver)

ax0.set_xlabel(r'coordinate ($\mathrm{\AA}$)')
ax0.set_xticks(np.linspace(-0.0, 1.5, 2))
ax0.set_xticks(np.linspace(-0.75, 2.25, 5), minor=True)
ax0.set_ylabel(r'$V$ (meV)')
ax0.set_ylim([-10,160])
ax0.set_yticks(np.linspace(0,150,4))
img = mpimg.imread("topview_marked.png")
ax1.imshow(img)
ax1.set_axis_off()

ax2.set_ylabel(r"$\Lambda \ (\mathrm{ps}^{-1})$")
ax2.set_xlabel(r"$\omega \  (\mathrm{eV})$")
ax3.set_xlabel(r"$\omega \  (\mathrm{eV})$")


df = pd.read_csv(aims / "16_kpts_friction.csv")
freqs = df.iloc[:,0].to_numpy()
Lambda = df.iloc[:,1:].to_numpy()
sigma = 0.01
for s, c, ax in zip(
    [0.02, 0.04],
    [mycolours.orange, mycolours.purple],
    [ax2, ax3]):
    broad_Lambda = np.zeros_like(Lambda)
    for L, bL in zip(Lambda.T, broad_Lambda.T):
        bL[:] = FrictionParser.broaden_friction_tensor_grid(None, s, freqs, L, grid_window=sigma)[1]
    ax.plot(freqs, broad_Lambda[:,-1], c=c, label='hcp')
    ax.plot(freqs, broad_Lambda[:,5], c=c, ls='--', label='bridge')
    ax.plot(freqs, broad_Lambda[:,0], c=c, ls=":", label='fcc')
    ax.set_xlim([0,0.6])
    ax.set_ylim([-0.1, 3.2])
    ax.set_yticks(np.linspace(0,3,4))
    ax.legend(
        labelspacing=0.2,
        bbox_transform=ax.transAxes, 
        loc='lower right', 
        handlelength=1.75,
        bbox_to_anchor=(1.03, -0.04))


lines = []
for T, N in zip(
    [50, 100, 150],
    [96,  48,  32]
):
    for model, c in zip(
        ['ohmic/pbe_light_frozen/H/wcut_4000',
         'sigma_40_meV/pbe_light_frozen/H/wcut_4000',
         'sigma_20_meV/pbe_light_frozen/H/wcut_4000'],
        [mycolours.green,  mycolours.purple,  mycolours.orange]
        ):
        etas, kQTST = get_kQTST_data(model, T, N)
        # scale by a factor of 3 to account for later comparison with expt
        line, = ax4.plot(etas, np.log10(3 * kQTST * u.str2base('1 ps')), c=c, ls=':', marker="+")
        if T == 50:
            lines.append(line)
ax4.text(0.94, 0.96, r'150 K',
        bbox={'pad': 2, 'facecolor': 'white'}, ha='right', va='top',
        transform=ax4.transAxes)
ax4.text(0.94, 0.61, r'100 K',
        bbox={'pad': 2, 'facecolor': 'white'}, ha='right', va='top',
        transform=ax4.transAxes)
ax4.text(0.94, 0.2, r' 50 K',
        bbox={'pad': 2, 'facecolor': 'white'}, ha='right', va='top',
        transform=ax4.transAxes)
ax4.set_ylabel(r'$\log_{10} [k_{\text{QTST}} (\mathrm{ps}^{-1})]$')
ax4.set_xlabel(r'friction scale factor')
ax4.legend(
    lines[::-1], 
    [r'$0.02~\mathrm{eV}$', r'$0.04~\mathrm{eV}$', 'Ohmic' ],
    labelspacing=0.2,
    bbox_transform=ax4.transAxes, 
    loc='lower left', 
    handlelength=1.75,
    bbox_to_anchor=(-0.01, -0.02))
ax4.set_ylim([-3.80, -1.00])
ax4.set_xlim([-0.75, 10.75])


fig.tight_layout()
text = []
for ax, l, x, y in zip(
    axes, string.ascii_lowercase,
    3*[0.085, 0.585],
    [0.965, 0.965, 0.67, 0.67, 0.42]):
    t = ax.text(
        x, y, f'({l})', transform=fig.transFigure, 
        ha='right', va='bottom', 
        clip_on=False)
    text.append(t)
fig.subplots_adjust(hspace=3.0, wspace=10.0, 
                    top=0.97, left=0.17, right=0.96,
                    bottom=0.08)
# move the image
l, b, w, h = ax1.get_position().bounds
ax1.set_position([l-0.05, b-0.03, w*1.2, h*1.2])
plt.savefig('fig1.png')
plt.savefig('fig1.eps')
