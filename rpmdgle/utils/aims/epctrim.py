#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   epctrim.py
@Time    :   2024/08/08 13:15:35
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   Script for zeroing out electron-phonon coupling matrix elements outside of a band of energies centred on the Fermi level
'''


from __future__ import print_function, division, absolute_import
from gtlib.utils.aims.friction import BaseParser, MPI
from gtlib.utils.aims.elsi import ELSIReader, ELSIWriter
from typing import Optional
from scipy.sparse import csc_array
import numpy as np
import logging


class EPCTrimmer(BaseParser):

    # given an energy range for calculating the friction tensor,
    # consider states at +/- this multiple of the largest item in the range
    # either side of the Fermi level
    energy_band = 2 # 2 is the default in FHI-aims
    
    def _get_energy_manifold_indices(
            self,
            E: np.ndarray,
            maxenergy: float,
        ) -> tuple[int, int, int, int] :
        # Last state below the energy cut-off
        lowest_energy = self.e_fermi - self.energy_band*maxenergy
        logging.debug(f"{self.e_fermi = }")
        logging.debug(f"{lowest_energy = }")
        lo_state = np.searchsorted(E, lowest_energy, side="left")-1
        logging.debug(f"{E[lo_state] = }")
        # First state above the energy cut-off
        highest_energy = self.e_fermi + self.energy_band*maxenergy
        logging.debug(f"{highest_energy = }")
        hi_state = np.searchsorted(E, highest_energy, side="right")
        logging.debug(f"{E[hi_state] = }\n")
        return lo_state, hi_state
    
    def trim(
            self,
            atom: int,
            cart: int,
            maxenergy: Optional[float] = 2.0,
            overwrite: Optional[bool] = True):
        
        spin_list, idx_list, _ = self.make_mpi_iterables(
            len(self.eigvals), np.arange(self.nk), self.spin)
        bands = np.reshape(self.eigvals, (len(spin_list), -1))
        start, end = self.get_mpi_slices(len(spin_list))
        for spin, idx, energies in zip(
                spin_list[start:end],
                idx_list[start:end], 
                bands[start:end]):
            # iterate over all the k-points for every spin channel
            print(f"{idx = }"f" {spin = }")
            k = idx + 1
            input_name = self.get_csc_file_name(
                k, atom, cart, spin if self.spin else None)
            reader: ELSIReader = ELSIReader(input_name)
            lo_state, hi_state = self._get_energy_manifold_indices(energies, maxenergy)
            col_slice = slice(lo_state, hi_state+1)
            indptr = np.zeros(reader.n_basis+1, dtype=int)
            row_indices = np.arange(reader.n_basis, dtype=int)[col_slice]
            local_nnz = len(row_indices)
            indices = []
            data = []
            for i_col, _ in enumerate(energies):
                if i_col < lo_state:
                    continue
                if i_col > hi_state:
                    indptr[i_col+1:] = indptr[i_col]
                    break
                data.append(reader.read_matrix_column(i_col)[col_slice])
                indices.append(row_indices.copy())
                indptr[i_col+1] = indptr[i_col] + local_nnz
            trimmed_epc = csc_array((
                np.concatenate(data),
                np.concatenate(indices),
                indptr), shape=(reader.n_basis, reader.n_basis))
            trimmed_epc.eliminate_zeros()
            if overwrite:
                output_name = input_name
            else:
                output_name = f"trimmed_{input_name}"
            logging.debug(f"{reader.nnz = }")
            logging.debug(f"{trimmed_epc.nnz = }")
            writer: ELSIWriter = ELSIWriter()
            writer.write_csc_matrix(output_name, trimmed_epc, reader.n_electrons)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero out electron-phonon coupling matrix elements outside of a band of energies centred on the Fermi level",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dir',
                        type=str,
                        help="Path to the directory containing the EPC files")
    parser.add_argument('-o', '--output',
                        type=str,
                        default='aims.out',
                        help='Name of the aims output file.')
    parser.add_argument('-p', '--spin-polarized',
                        action='store_true',
                        help='Use when the calculation was run with `spin collinear`.')
    parser.add_argument("--atom", type=int, default=1, 
                       help="Index of the atom from which to trim the matrix")
    parser.add_argument("--cart", type=int, default=1, choices=[1,2,3],
                       help="Cartesian component of atom coordinate for which to trim the matrix.")
    parser.add_argument("--keep", action="store_true",
                       help="Keep the original EPC file, writing the trimmed matrix to a *.csc file with the prefix `trimmed_`.")
    parser.add_argument("--max-energy", type=float, default=2.0,
                       help="Trim the matrices, such that subsequent post-processing would give unchanged spectral densities up to this energy (in eV).")
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        choices=[0,1,2],
                        help="Verbosity level, from silent to debug.")
    
    args: argparse.Namespace = parser.parse_args()
   
    # Initialize MPI environment
    is_mpi_launched = MPI.Is_initialized()
    if is_mpi_launched:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = (rank == 0)
    else:
        comm = None
        rank = 0
        size = 1
        root = True

    logger = logging.getLogger(__name__)
    if root:
        if args.verbosity == 2:
            level = logging.DEBUG
        elif args.verbosity == 1:
            level = logging.INFO
        else:
            level = logging.ERROR
    else:
        level = logging.ERROR
    
    logging.basicConfig(
        level=level,
        format="%(message)s")
    
    epc_trimmer: EPCTrimmer = EPCTrimmer(
        args.dir, args.output, spin_collinear=args.spin_polarized,
        comm=comm, root=root, size=size, rank=rank)
    
    epc_trimmer.trim(
        args.atom, args.cart, maxenergy=args.max_energy, overwrite=(not args.keep))
    

if __name__ == "__main__":
    main()