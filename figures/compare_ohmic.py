#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   figure01.py
@Time    :   2024/10/22 10:49:18
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   Compare the Ohmic friction constant derived from broadened kernels using different widths of the Gaussian window.
'''


from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')
import matplotlib.image as mpimg
from pathlib import Path
from rpmdgle.pes.splined import OneDCubic as Splined
from rpmdgle.units import SI
from eftools.friction import FrictionParser
import json
import numpy as np
import mycolours
import pandas as pd
import glob
import string
import os

data = Path(os.environ['RPMDGLE']) / 'data'
aims = Path(os.environ['RPMDGLE']) / 'abinitio' / 'pbe_light_frozen' / 'slab+H' / '06_layers'

fig, ax = plt.subplots(figsize=(3.375, 2.1))

df = pd.read_csv(aims / "16_kpts_friction.csv")
freqs = df.iloc[:,0].to_numpy()
Lambda = df.iloc[:,1:].to_numpy()
sigma = 0.01
broad_sigmas = np.arange(1.0, 10.0)/10
cmap = plt.get_cmap("viridis")
colors = [cmap(x) for x in np.linspace(0.1, 0.9, len(broad_sigmas))]
for bs, c in zip(broad_sigmas, colors):
    broad_Lambda = np.zeros_like(Lambda)
    for L, bL in zip(Lambda.T, broad_Lambda.T):
        bL[:] = FrictionParser.broaden_friction_tensor_grid(None, bs, freqs, L, grid_window=sigma)[1]
    ax.plot(freqs, broad_Lambda[:,-1], c=c, label=f'{bs:.1f}')
    # ax.plot(freqs, broad_Lambda[:,5], c=c, ls='--', label='bridge')
    # ax.plot(freqs, broad_Lambda[:,0], c=c, ls=":", label='fcc')
    ax.set_xlim([0,0.6])
    ax.set_ylim([-0.1, 3.2])
    # ax.set_yticks(np.linspace(0,3,4))
    # ax.legend(
    #     labelspacing=0.2,
    #     bbox_transform=ax.transAxes, 
    #     loc='lower right', 
    #     handlelength=1.75,
    #     bbox_to_anchor=(1.03, -0.04))
ax.legend()
plt.show()