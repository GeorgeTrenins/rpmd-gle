#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   parse_energy.py
@Time    :   2024/08/13 15:58:55
@Author  :   George Trenins
@Desc    :   Read all the converged system energies from an aims.out file 
'''


from __future__ import print_function, division, absolute_import
import argparse
from pathlib import Path
from typing import Union
import numpy as np

def main(output_file: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import re

    total_pattern = r"total\s+energy\s*:\s*(-?\d+\.\d+)\s*ha\s*(-?\d+\.\d+)\s*ev"
    zerotemp_pattern = r"total\s+energy,\s+t\s+->\s+0\s*:\s*(-?\d+\.\d+)\s*ha\s*(-?\d+\.\d+)\s*ev"
    free_pattern = r"electronic\s+free\s+energy\s*:\s*(-?\d+\.\d+)\s*ha\s*(-?\d+\.\d+)\s*ev"
    converged_pattern = r"self-consistency\s+cycle\s+converged"

    total_cache = [None]
    total_energies = []
    zero_cache = [None]
    zerotemp_energies = []
    free_cache = [None]
    free_energies = []

    with open(output_file, 'r') as f:
        for line in f:
            for pattern, cache in zip(
                    [total_pattern, zerotemp_pattern, free_pattern],
                    [total_cache, zero_cache, free_cache]):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    cache[:] = [float(match.group(2))]
            match = re.search(converged_pattern, line, re.IGNORECASE)
            if match:
                for energies, cache in zip(
                        [total_energies, zerotemp_energies, free_energies],
                        [total_cache, zero_cache, free_cache]):
                    value = cache[0]
                    assert value is not None, "Corrupted aims.out"
                    energies.append(value)
                    cache[:] = [None]
    
    return (np.asarray(total_energies),
            np.asarray(zerotemp_energies),
            np.asarray(free_energies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the otput of FHI-aims "
                                    "for elelctronic friction calculations.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('dir',
                        type=str,
                        help="Path to the directory containing the output files")
    parser.add_argument('-o', '--output',
                        type=str,
                        default='aims.out',
                        help='Name of the aims output file.')
    args = parser.parse_args()
    tot, zero, free = main(Path(args.dir) / args.output)
    print(f"{np.c_[tot, zero, free]}")
