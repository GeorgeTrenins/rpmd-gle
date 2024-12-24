#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   figure01.py
@Time    :   2024/10/22 10:49:18
@Author  :   George Trenins
@Desc    :   Compare convergence of transmission coefficients, 
'''


from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')
from pathlib import Path
import mycolours
import json
import numpy as np
import string
import matplotlib as mpl
import os
from rpmdgle.rates.analyse import fancy_flux
from rpmdgle import units

u = units.eVAamu()

def get_kappa(model, T, N, eta):
    root = (Path(os.environ['RPMDGLE']) / 'data' / model /
            f"{T:03d}K" / f"nbeads_{N:03d}" / f"eta_{eta:05.2f}" )
    t, kappa = np.loadtxt(root / "c_fs.csv", unpack=True, usecols=(0,3))
    with open(root / 'kQTST.json', 'r') as f:
        tst_data = json.load(f)
    UNITS = getattr(units, tst_data["UNITS"])()
    dt = UNITS.str2base(f'{t[0]} ps')
    kTST_hcp = tst_data["kQTSTP"]
    kTST_fcc = tst_data["kQTST"]
    fancy_cfs = fancy_flux(dt, kappa, kTST_hcp, kTST_fcc)
    # average over statistical fluctuations once the plateau is reached:
    if int(eta) == 1:
        # takes longer for weak coupling
        mask = t > 2.0
    elif int(eta) == 10:
        # onset is quick for strong damping
        mask = np.logical_and(t > 1.0, t<1.5)
    else:
        raise NotImplementedError
    return np.mean(fancy_cfs[mask]) / kTST_hcp


fig = plt.figure(figsize=(3.375, 1.9))

ax0 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
ax1 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1, sharex=ax0, sharey=ax0)
plt.setp(ax1.get_yticklabels(), visible=False)
ax0.set_xticks(np.linspace(0,1.5,4))

T = 200
model = 'sigma_20_meV/pbe_light_frozen/H/wcut_4000'
nbath = [64, 128, 256, 512]
reds = [mpl.colormaps['Reds'](v) for v in np.linspace(0.3,1.0,len(nbath))]
for ax, eta in zip((ax0, ax1), [1, 10]):
    lines = []
    for c, nb in zip(reds, nbath):
        t, cfs = np.loadtxt(Path(os.environ['RPMDGLE']) / 'data' /
                            model / f"{T:03d}K" / "nbeads_001" /
                            f"eta_{eta:05.2f}" / f"nbath_{nb:04d}" / "c_fs.csv", 
                            unpack=True, usecols=(0,3))
        mask = t < 1.5
        line, = ax.plot(t[mask], cfs[mask], c=c, lw=1)
        lines.append(line)
    t, cfs = np.loadtxt(Path(os.environ['RPMDGLE']) / 'data' /
                        model / f"{T:03d}K" / "nbeads_001" /
                        f"eta_{eta:05.2f}" / "c_fs.csv", 
                        unpack=True, usecols=(0,3))
    mask = t < 1.5
    line, = ax.plot(t[mask], cfs[mask], c='#00def3', lw=1)
    lines.append(line)
    ax.set_ylim([-0.5, 1.1])
    ax.set_xlabel('time (ps)')

ax0.set_ylabel('transmission coefficient')
ax0.set_xlim([-0.1, 1.6])
leg = ax1.legend(lines,
           [rf'${n}$' for n in nbath]+[r'aux'],
           ncol=5, #(len(nbath)//2+1),
           bbox_transform=fig.transFigure, 
           loc='upper center', 
           handlelength=1.75,
           bbox_to_anchor=(0.5, 1.0))


fig.subplots_adjust(
    hspace=0.8, wspace=0.15, left=0.175, right=.95, top=0.83, bottom=0.21)
ax0.yaxis.set_label_coords(-0.325, 0.45)

for ax, l in zip(
    [ax0, ax1], string.ascii_lowercase):
    t = ax.text(
        0.95, 0.9, f'({l})', transform=ax.transAxes, 
        ha='right', va='top', 
        clip_on=False)
fig.savefig('fig2.png')
fig.savefig('fig2.eps')