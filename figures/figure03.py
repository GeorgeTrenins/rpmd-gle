#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   figure01.py
@Time    :   2024/10/22 10:49:18
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   Compare tunnelling crossover between different friction models + comparison of best model to HeSE
'''


from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')
from pathlib import Path
from rpmdgle.units import eVAamu
import mycolours
import json
import numpy as np
import string
import os
from rpmdgle.rates.analyse import fancy_flux
from rpmdgle import units


fig, (ax0, ax1) = plt.subplots(figsize=(3.375, 3.5), nrows=2, sharex=True)
u = eVAamu()

T_lst = np.asarray([50, 60, 70, 80, 90, 100, 125, 150, 160, 175, 200, 225, 250, 300])
N_H = np.asarray([96, 64, 64, 48, 48, 48, 32, 32, 24, 24, 24, 24, 16, 16])
N_D = np.asarray([64, 48, 48, 32, 32, 32, 24, 24, 16, 16, 16, 16, 12, 12])

def get_rate(model, T, N, eta):
    root = (Path(os.environ['RPMDGLE']) / 'data' / model /
            f"{T:03d}K" / f"nbeads_{N:03d}" / f"eta_{eta:05.2f}" )
    try:
        t, kappa = np.loadtxt(root / "c_fs.csv", unpack=True, usecols=(0,3))
    except:
        return np.nan, np.nan
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
    # scale rates by a factor of three for comparison with expt
    return (kTST_hcp * u.str2base('1 ps') * 3,
            np.mean(fancy_cfs[mask]) * u.str2base('1 ps') * 3)

ohmic_rpmd_h_01 = []
ohmic_rpmd_h_01_tst = []
eta = 1.0
model = "ohmic/pbe_light_frozen/H/wcut_4000"
for T, N in zip(T_lst, N_H):
    kTST, k = get_rate(model, T, N, eta)
    ohmic_rpmd_h_01_tst.append(kTST)
    ohmic_rpmd_h_01.append(k)

ohmic_cl_h_01 = []
for T in T_lst:
    ohmic_cl_h_01.append(get_rate(model, T, 1, eta)[1])

model = "ohmic/pbe_light_frozen/D/wcut_4000"
ohmic_rpmd_d_01 = []
ohmic_rpmd_d_01_tst = []
for T, N in zip(T_lst, N_D):
    kTST, k = get_rate(model, T, N, eta)
    ohmic_rpmd_d_01_tst.append(kTST)
    ohmic_rpmd_d_01.append(k)

ohmic_cl_d_01 = []
for T in T_lst:
    ohmic_cl_d_01.append(get_rate(model, T, 1, eta)[1])

model = "sigma_20_meV/pbe_light_frozen/H/wcut_4000"
super_rpmd_h_01 = []
eta = 1.0
for T, N in zip(T_lst, N_H):
    super_rpmd_h_01.append(get_rate(model, T, N, eta)[1])

super_cl_h_01 = []
for T in T_lst:
    super_cl_h_01.append(get_rate(model, T, 1, eta)[1])

model = "sigma_20_meV/pbe_light_frozen/D/wcut_4000"
super_rpmd_d_01 = []
for T, N in zip(T_lst, N_D):
    super_rpmd_d_01.append(get_rate(model, T, N, eta)[1])

super_cl_d_01 = []
for T in T_lst:
    super_cl_d_01.append(get_rate(model, T, 1, eta)[1])


Tinv = 1000/np.asarray(T_lst)
    
x, rate = np.loadtxt( 
    Path(os.environ["RPMDGLE"]) / 'data' / 'townsend_cu111_H_rates.csv', 
    unpack=True, delimiter=',', skiprows=1)
expt, = ax0.plot(x, np.log10(rate), 'x', ms=4, 
         mfc='k', mec='k')
qtst, = ax0.plot(Tinv, np.log10(ohmic_rpmd_h_01_tst), 'k-.')
rpmd_super, = ax0.plot(Tinv, np.log10(super_rpmd_h_01), '^', ms=4,
          mec=mycolours.orange, mfc=mycolours.orange)

# add straight-line fit
slc=slice(-4,-1,None)
xdata = Tinv[slc]
ydata = np.log10(super_rpmd_h_01)[slc]
m, c = np.polyfit(xdata, ydata, 1)
grid = np.linspace(3,8,100)
ax0.plot(grid,  m*grid + c, c='0.7', lw=1, zorder=-5)

rpmd_ohmic, = ax0.plot(Tinv, np.log10(ohmic_rpmd_h_01), '^', ms=4,
          mec=mycolours.green, mfc=mycolours.green)
cl_super, = ax0.plot(Tinv, np.log10(super_cl_h_01), c=mycolours.orange, ls=":")
cl_ohmic, = ax0.plot(Tinv, np.log10(ohmic_cl_h_01), c=mycolours.green, ls=":")

# add deuterium
x, rate = np.loadtxt( 
    Path(os.environ["RPMDGLE"])  / 'data' / 'townsend_cu111_D_rates.csv', 
    unpack=True, delimiter=',', skiprows=1)
ax1.plot(x, np.log10(rate), 'x', ms=4,
        mec='k', mfc='k')
ax1.plot(Tinv, np.log10(ohmic_rpmd_d_01_tst), 'k-.')
ax1.plot(Tinv, np.log10(ohmic_rpmd_d_01), '^', ms=4,
         mec=mycolours.green, mfc=mycolours.green)

ax1.plot(Tinv, np.log10(ohmic_cl_d_01), c=mycolours.green, ls=":")
ax1.plot(Tinv, np.log10(super_rpmd_d_01), '^', ms=4,
         mec=mycolours.orange, mfc=mycolours.orange)

# add straight-line fit
slc=slice(-4,-1,None)
xdata = Tinv[slc]
ydata = np.log10(super_rpmd_d_01)[slc]
m, c = np.polyfit(xdata, ydata, 1)
grid = np.linspace(3,8,100)
ax1.plot(grid,  m*grid + c, c='0.7', lw=1, zorder=-5)
ax1.plot(Tinv, np.log10(super_cl_d_01), c=mycolours.orange, ls=":")

ax0.set_xlim([3.7,7.1])
ax0.set_ylim([-2.45, -0.75])
ax1.set_ylim([-3.25, -1.25])
for ax in [ax0, ax1]:
    ax.set_xticks(list(range(4,8)))
for ax in [ax0, ax1]:
    ax.set_ylabel(r"$\log_{10} [k(1/\mathrm{ps})] $")
for ax, l in zip(
    [ax0, ax1], string.ascii_lowercase):
    t = ax.text(
        0.95, 0.92, f'({l})', transform=ax.transAxes, 
        ha='right', va='top', 
        clip_on=False)
ax1.set_xlabel(r"$1000 / T$ (1/K)")  
    
legend = ax0.legend(
    [cl_ohmic, cl_super, qtst, rpmd_ohmic, rpmd_super, expt],
    [
        'classical (Ohmic)', r'classical (0.02 eV)', 'QTST',
        'RPMD (Ohmic)', r'RPMD (0.02 eV)', 'experiment'
    ], 
    ncol=2,
    bbox_transform=ax0.transAxes, 
    loc='lower center', 
    bbox_to_anchor=(0.5, 1.02),
    handletextpad=0.2)

fig.subplots_adjust(
    hspace=0.12, wspace=0.35, left=0.175, right=.95, top=0.81)    
fig.savefig('fig3.png')
fig.savefig('fig3.eps')