#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   friction.py
@Time    :   2024/04/24 10:29:48
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   Process electronic friction output of aims
'''


from __future__ import print_function, division, absolute_import
from gtlib.utils import grep
from gtlib.utils.sysbath import kernel_transform
import numpy as np
import re
import pandas as pd
import numpy as np
import itertools
import sys
from scipy.special import erf
from gtlib.units import atomic
from pathlib import Path
from gtlib.utils.aims.elsi import ELSIReader
from typing import Optional, Union
from scipy.ndimage import gaussian_filter1d
from ase.io.aims import read_aims_output
from ase.atoms import Atoms
try:
    from mpi4py import MPI
except ImportError:
    from gtlib.utils import MPI


class BaseParser(object):
    UNITS = atomic()
    # The values below are the same as in FHI-aims 231208
    # Only change if you know what you're doing
    hartree = 27.2113845    # 1 Hartree in eV
    bohr = 0.5291772        # 1 Bohr in angstrom
    ps = 41341.37333656136  # 1 ps in atomic units of time
    
    def __init__(self, 
                 dir: Union[str, Path],
                 aimsout: Optional[str] = "aims.out",
                 spin_collinear: Optional[bool] = False,
                 comm: Optional[MPI.Intracomm] = None,
                 root: Optional[bool] = True,
                 size: Optional[int] = 1,
                 rank: Optional[int] = 0
                ) -> None:
        
        self.comm = comm
        self.root = root
        self.size = size
        self.rank = rank
        self.spin = spin_collinear
        self.aimsdir: Path = Path(dir)
        self.aimsout: Path = self.aimsdir / aimsout
        # Parse Fermi level
        if root:
            self.e_fermi = self.parse_fermi_level()
        else:
            self.e_fermi = None
        if comm:
            self.e_fermi = self.comm.bcast(self.e_fermi, root=0)
        # Number of k-points is determined while parsing the KS eigenvalues
        self.nk = None 
        # Parse KS eigenvalues
        if root:
            eigvals: list[np.ndarray] = []
            if spin_collinear:
                for i in [1,2]:
                    eigvals.append(self.parse_eigvals(i))
            else:
                eigvals.append(self.parse_eigvals())
            self.eigvals = np.asarray(eigvals)
            shape_eigvals = self.eigvals.shape
        if comm:
            if not root:
                shape_eigvals = None
            shape_eigvals = self.comm.bcast(shape_eigvals, root=0)
            self.nk = self.comm.bcast(self.nk, root=0)
            if not root:
                self.eigvals = np.empty(shape_eigvals, dtype=float)
            self.comm.Bcast(self.eigvals, root=0)

    @staticmethod
    def make_mpi_iterables(
            n_spin_channel: int,
            k_weights: np.ndarray,
            spin: bool) -> tuple[list[int], list[int], list[float]]:
        spin_list = []
        idx_list = []
        kw_list = []
        for s, (idx, kw) in itertools.product(
                range(n_spin_channel),
                enumerate(k_weights)):
            spin_list.append(s+1 if spin else None)
            idx_list.append(idx)
            kw_list.append(kw)
        return spin_list, idx_list, kw_list
    
    def get_mpi_slices(self, ntasks: int) -> tuple[int, int]:
        chunk_size = ntasks // self.size
        remainder = ntasks % self.size
        if self.rank < remainder:
            start = self.rank * (chunk_size + 1)
            end = start + chunk_size + 1
        else:
            start = self.rank * chunk_size + remainder
            end = start + chunk_size
        return start, end
   
    def get_csc_file_name(
        self,
        k : int,
        atom : int,
        cart : int,
        spin : Optional[int] = None)-> str:

        if spin is None:
            if self.spin:
                raise RuntimeError('Must specify the spin for parsing coupling elements from a collinear calculation')
            else:
                fmask = f'epc_atom_{atom:06d}_cart_{cart:01d}_k_{k:06d}.csc'
        else:
            fmask = f'epc_atom_{atom:06d}_cart_{cart:01d}_k_{k:06d}_spin_{spin:01d}.csc'
        return self.aimsdir / fmask


    def parse_fermi_level(self) -> float:
        fermi_level_str = grep.grep_context(
            'Self-consistency cycle converged',
            self.aimsout,
            before=21,
            after=0,
            head=1)[0]

        pattern = r'Chemical Potential .* :(.*) eV'
        match = re.search(pattern, fermi_level_str)
        if match is None:
            raise RuntimeError(f"Could not parse the Fermi level from {self.aimsout}")
        return float(match.group(1))
    
    def parse_eigvals(self, i_spin: Optional[int]=None) -> np.ndarray:
        if i_spin is None or i_spin == 1:
            start_idx = 3
            self.nk = 0
        elif i_spin == 2:
            start_idx = 4 + self.nk
        else:
            raise RuntimeError(f'Expecting spin = 1 or 2, instead got {i_spin=}')
        evals = []
        with open(self.aimsdir / 'friction_KS_eigenvalues.out', 'r') as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                try:
                    data = np.asarray(line.split()[1:], dtype=float)
                except ValueError:
                    if line.split()[:2] == ["Spin", "component"]:
                        break
                    else:
                        raise
                evals.append(data)
                if i_spin != 2:
                    self.nk += 1
        return np.asarray(evals)


class FrictionParser(BaseParser):

    # given an energy range for calculating the friction tensor,
    # consider states at +/- this multiple of the largest item in the range
    # either side of the Fermi level
    energy_band = 2 # 2 is the default in FHI-aims
    # largest occupation number to consider for the 
    # high-energy manifold
    upper_band_occ = 0.9999 # 0.9999 is the default in FHI-aims
    # smallest occupation number to consider for the 
    # low-energy manifold
    lower_band_occ = 0.001 # 0.001 is the default in FHI-aims
    # energy cut-off, in numbers of standard deviation from the mean,
    # beyond which contributions to nascent delta functions are discarded
    # to avoid underflow
    nsigma = 6

    def __init__(self, 
                 dir: Union[str, Path],
                 aimsout: Optional[str] = "aims.out",
                 spin_collinear: Optional[bool] = False,
                 comm: Optional[MPI.Intracomm] = None,
                 root: Optional[bool] = True,
                 size: Optional[int] = 1,
                 rank: Optional[int] = 0
                ) -> None:
        
        super().__init__(
            dir, aimsout=aimsout, spin_collinear=spin_collinear, 
            comm=comm, root=root, size=size, rank=rank)
        
        self.kgrid: np.ndarray
        self.kweights: np.ndarray

        # Parse k-grid
        if root:
            self.kgrid, self.kweights = self.parse_kgrid()
            shape_kg = self.kgrid.shape
            shape_kw = self.kweights.shape
        if not root:
            shape_kg = None
            shape_kw = None
        if comm:
            shape_kg = self.comm.bcast(shape_kg, root=0)
            if not root:
                self.kgrid = np.empty(shape_kg, dtype=float)
            self.comm.Bcast(self.kgrid, root=0)
            shape_kw = self.comm.bcast(shape_kw, root=0)
            if not root:
                self.kweights = np.empty(shape_kw, dtype=float)
            self.comm.Bcast(self.kweights, root=0)
        
        # Parse masses
        if root:
            with open(self.aimsout, 'r') as f:
                cell: Atoms = read_aims_output(f, index=-1)
            m = np.asarray(cell.get_masses() * self.UNITS.amu, dtype='d')
        if comm:
            if root:
                len_m = len(m)
            else:
                len_m = None
            len_m = comm.bcast(len_m, root=0)
            if not root:
                m = np.empty(len_m, dtype='d')
            comm.Bcast(m, root=0)
        self.masses = np.asarray(m, dtype=float)

        # Parse the width of the window used to estimate the spectral density
        if root:
            try:
                ans = grep.grep('friction_window_size', self.aimsdir / 'control.in')
            except FileNotFoundError:
                ans = []
            if len(ans) == 0:
                grid_window = 0.01 # eV
            elif len(ans) == 1:
                grid_window = float(ans[0].split()[1])
            else:
                raise RuntimeError(f"Got multiple entries for 'friction_window_size': {ans}")
        else:
            grid_window = None
        if comm:
            grid_window = comm.bcast(grid_window, root=0)
        self.grid_window = grid_window

    @staticmethod
    def broaden_friction(
            tensor_grid: np.ndarray, 
            energies: np.ndarray,
            bandwidth: float,
            axis: Optional[int] = -1) -> np.ndarray:
        h = (energies[-1] - energies[0])/(len(energies)-1)
        return gaussian_filter1d(
            tensor_grid, 
            sigma=bandwidth/h, 
            mode='constant', 
            cval=0.0, 
            truncate=6.0,
            axis=axis
        )

    @staticmethod    
    def F_D_occupation(
            beta: float,
            e_fermi: float,
            energy : Union[float, np.ndarray]) -> Union[float, np.ndarray] :
        # get dimensionless energies
        beta_e_fermi = beta * e_fermi
        beta_energy = beta * energy
        diff = beta_energy - beta_e_fermi
        # to avoid overflow
        cutoff = 100
        smaller = diff < -cutoff
        greater = diff > cutoff
        bool_arr = np.logical_or(smaller, greater)
        tmp = np.where(bool_arr, 0.0, diff)
        n = np.where(bool_arr, 1.0, 1/(np.exp(tmp) + 1))
        n[greater] = 0.0
        return n
    
    @staticmethod
    def gaussian_delta(x, mean, sdev, nsigma=6):
        y = (x-mean)/sdev
        ans = np.zeros_like(y)
        bool_arr = np.abs(y) < nsigma
        ans[bool_arr] = np.exp(-y[bool_arr]**2/2)
        ans /= np.sqrt(2*np.pi) * sdev
        return ans, bool_arr
    
    @staticmethod
    def erf_step(mean, sdev, nsigma=6, upper=None):
        """Error-function representation of the step function that corresponds
        to the integral of gaussian_delta() from 0 to infinity

        Args:
            mean (np.ndarray)
            sdev (np.ndarray)
            nsigma (int, optional): Number of standard deviations from zero, beyond which to set the answer to 0 or 1. Defaults to 6.
            upper (float, optional): if specified, gives the error-function representation of a top-hat
            function.
        """

        y = mean/sdev
        ans = np.zeros_like(y)
        mask = np.logical_and(y > -nsigma, y < nsigma)
        ans[mask] = 0.5 * (1 + erf(y[mask]/np.sqrt(2)))
        mask = (y >= nsigma)
        ans[mask] = 1.0
        if upper:
            z = upper/sdev
            assert z > nsigma, "The upper edge of the nascent tophat function is too close to the origin"
            mask = np.logical_and(y > z-nsigma, y < z+nsigma)
            ans[mask] = 0.5 * (1 + erf( -(y[mask]-z)/np.sqrt(2)) )
        return ans, (y > -nsigma)

    def _fdiff_by_omega(
            self, 
            beta: float,
            f: np.ndarray,
            fdiff: np.ndarray, 
            Omega: np.ndarray) -> np.ndarray:
        bw = beta*Omega
        small = np.abs(bw) < 0.005
        tmp = np.where(small, 1, Omega)
        neg_f = 1 - f
        ans = np.where(
            small,
            beta*f*neg_f * (1 + bw/2 * (
                    f - neg_f + bw/3 * (1 - 6*f*neg_f))),
            fdiff / tmp)
        return ans
    
    def _get_energy_manifold_indices(
            self,
            E: np.ndarray,
            occ: np.ndarray,
            maxfreq: float,
        ) -> tuple[int, int, int, int] :
        e_fermi = self.e_fermi / self.hartree
        lowest_energy = e_fermi - self.energy_band*maxfreq
        # Last state below the energy cut-off
        lo_state = np.searchsorted(E, lowest_energy, side="left")-1
        # First state above occupation cut-off
        lumo_state = np.searchsorted(-occ, -self.upper_band_occ, side="left")
        highest_energy = e_fermi + self.energy_band*maxfreq
        # First state above the energy cut-off
        hi_state = np.searchsorted(E, highest_energy, side="right")
        # Lowest state at or below occupation cut-off
        homo_state = np.searchsorted(-occ, -self.lower_band_occ, side="left")-1
        if homo_state > hi_state:
            homo_state = hi_state-1
        return lo_state, lumo_state, homo_state, hi_state

    def parse_kgrid(self) -> tuple[np.ndarray, np.ndarray]:
        k_info = grep.grep_context(
            'Using symmetry for reducing the k-points',
            self.aimsout,
            before=0,
            after=1,
            head=1)[0]
        num_k_pts = int(k_info.split()[-1])
        if num_k_pts != self.nk:
            raise ValueError(f"Error parsing k-grid, expected {self.nk} k-points, got {num_k_pts}.")
        regf = '[+-]?\d*\.?\d+'
        pattern = rf'.*k-point:\s+\d+ at\s+({regf})\s+({regf})\s+({regf})\s+, weight:\s+({regf})'
        prog = re.compile(pattern)
        k_info = grep.grep_context(
            '\s+\| k-point:\s+1.*',
            self.aimsout,
            before=0,
            after=num_k_pts-1, 
            head=1)[0]
        k_info = k_info.split('\n')[:-1]
        kgrid = []
        kweights = []
        for line in k_info:
            result = prog.match(line)
            if result is None:
                raise RuntimeError(f"Could not find k-grid information in {self.aimsout}. Have you included the directive `output k_point_list` in your control file?")
            kgrid.append(list(result.group(1,2,3)))
            kweights.append(result.group(4))
        return np.asarray(kgrid, dtype=float), np.asarray(kweights, dtype=float)
    
    def parse_friction_tensor(self) -> np.ndarray:
        data = []
        with open(self.aimsdir / 'friction_tensor.out', 'r') as f:
            for i, line in enumerate(f):
                if i == 0 or i%2 == 1:
                    continue
                else:
                    data.append(line.split())
        return np.asarray(data, dtype=float)
    
    def parse_friction_tensor_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nhead = 3
        data = []
        with open(self.aimsdir / 'friction_tensor_grid.out', 'r') as f:
            for i, line in enumerate(f):
                if i == nhead: 
                    break
                data.append(line.split()[-1])
        ndim = int(data[0])   # number of rows/columns in friction tensor
        nelem = (ndim*(ndim+1)) // 2
        nbin = int(data[2])   # number of points on the energy grid
        
        keep_pattern = r"\s+0"
        prog = re.compile(keep_pattern)
        def filtered_line_generator(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if prog.match(line):
                        yield [float(f) for f in line.split()]

        # Use pandas to read the CSV file from the filtered generator function
        df = pd.DataFrame(filtered_line_generator(self.aimsdir / 'friction_tensor_grid.out'))
        data = df.to_numpy().T
        energy, Re, Im = (np.reshape(arr, (nelem,nbin)) for arr in data)
        return energy[0], Re, Im
    
    def broaden_friction_tensor_grid(
            self,
            sigma: float,
            energy: np.ndarray,
            kernel: np.ndarray,
            grid_window: Optional[float] = None
            ) -> tuple[np.ndarray, np.ndarray]:
        """Recalculate the spectral density using a broader windowing function

        Args:
            * sigma (float): target window in eV, must be larger than the original window
            * energy (np.ndarray, shape(M,)): energy values, in eV, for which the spectral density is evaluated
            * kernel (np.ndarray, shape(...,M,)): spectral density/densities
            * grid_window (float, optional): the width of the Gaussian window originally used on the input data; by default use the value from `control.in`

        Returns:
            tuple[np.ndarray, np.ndarray]: energy and broadened spectral density
        """
        from scipy.ndimage import gaussian_filter1d

        energy = np.asarray(energy)
        kernel = np.asarray(kernel)
        assert energy.ndim == 1
        assert energy.shape[-1] == kernel.shape[-1]
        hh = (energy[-1] - energy[0])/(len(energy)-1)
        ans = np.zeros_like(kernel)
        flat_kernel = np.reshape(kernel, (-1, len(energy)))
        flat_ans = np.reshape(ans, (-1, len(energy)))
        if grid_window is None:
            grid_window = self.grid_window
        if sigma < grid_window:
            raise ValueError(f"The requested window cannot be narrower than the starting point of {grid_window} eV")
        extra_width = np.sqrt((sigma-grid_window)*(sigma+grid_window))
        reduced_width = extra_width / hh
        for i, row in enumerate(flat_kernel):
            flat_ans[i,:] = gaussian_filter1d(row, reduced_width, truncate=6, mode='mirror')
        return energy, ans
 
              
    def compute_tensor_element(
        self,
        atom1: int,
        cart1: int,
        atom2: int,
        cart2: int,
        grid_bound: float,
        grid_step: float,
        lower_grid_bound: Optional[float] = 0.0,
        domain: Optional[str] = 'fourier',
        T: Optional[float] = 300.0,
        freq_broadening: Optional[float] = 0.01,
        excitation_cutoff: Optional[float] = 2.00,
        mode: Optional[int] = 2,
        add_diag: Optional[bool] = False
    ):
        """Compute the element of the friction tensor that couples the momentum components
        `cart1` and `cart2` of `atom1` and `atom2`, respectively. The result is calculated in one of three possible "domains":
           * `fourier` will return the spectral density, multiplied by frequency (i.e., the Fourier transform of the memory friction kernel)
           * `laplace` will return the Laplace transform of the memory-friction kernel
           * `time` will return the memory-friction kernel itself (i.e., the representation in the time domain.)

        Args:
            * atom1, cart1, atom2, cart2 (int): see above
            * grid_bound (float): upper bound of the regular grid of energies (domain == `fourier`, `laplace`) or times (domain == `time`) for which the kernel is calculated. The energy must be specified in eV, and time in ps.
            * grid_step (float): energy/time increment of the regular grid.
            mode (Optional[str], optional): see above. Defaults to 'fourier'.
            * lower_grid_bound (float, optional): only considered for the Fourier domains; uses a custom value for the lower bound of the frequency grid (default is 0.0)
            * domain (str, optional): see above, default is 'fourier'
            * T (Optional[float], optional): electronic temperature in Kelvin. Defaults to 300.0.
            * freq_broadening (Optional[float], optional): standard deviation (width) of the Gaussians used as nascent Deltas in the calculation of spectral densities. Defaults to 0.01 eV.
            * excitation_cutoff (Optional[float], optional): Maximum excitation energy included in the cosine series expansion of the friction kernel. Defaults to 2 eV.
            mode (int, optional): Use the FHI-aims implementation, including restriction to strictly positive excitation energies and renormalisation of nascent deltas.
            * mode (int, optional): `0` corresponds to original FHI-aims implementation, `1` is the first fix, which extends the lower portion of the energy grid a few broadening widths below zero, and `2` is the final fix, which uses an analytical expression to include de-excitations.
            * add_diag (bool, optional): include diagonal coupling terms from the calculation of the friction (debugging only, DON'T DO THIS for production calculations); default is False.
        """

        beta = self.UNITS.betaTemp(T)
        assert domain in {'fourier', 'laplace', 'time'}, f"Mode must be one of ['fourier', 'laplace', 'time'], instead given {domain = }."
        assert grid_bound > 0, f"Upper bound of the grid must be positive, instead {grid_bound =}"
        assert grid_step > 0, f"Grid step must be positive, instead {grid_step =}"
        assert grid_step < grid_bound, f"Grid step must be smaller than the upper grid bound, instead {grid_step =} and {grid_bound =}"
        # construct grid
        grid = np.arange(lower_grid_bound, grid_bound + grid_step, grid_step)
        keep = grid < grid_bound + 0.001*grid_step
        # convert everything to atomic units
        grid = grid[keep] * (self.ps if domain == 'time' else 1/self.hartree)
        freq_broadening = freq_broadening / self.hartree
        excitation_cutoff = excitation_cutoff / self.hartree
        # prepare summation weights and indices for parallel computation
        spin_weight = 1 if self.spin else 2
        spin_list, idx_list, kw_list = self.make_mpi_iterables(
            len(self.eigvals), self.kweights, self.spin)
        bands = np.reshape(self.eigvals, (len(spin_list), -1))
        start, end = self.get_mpi_slices(len(spin_list))
        partial_kernel = np.zeros_like(grid, dtype=complex)
        for spin, idx, kw, band_energy in zip(
                spin_list[start:end],
                idx_list[start:end], 
                kw_list[start:end],
                bands[start:end]):
            # iterate over all the k-points for every spin channel
            print(f"{idx = }"f" {spin = }")
            k = idx + 1
            energies = band_energy / self.hartree       
            occ = self.F_D_occupation(beta, self.e_fermi/self.hartree, energies)
            reader1: ELSIReader = ELSIReader(self.get_csc_file_name(
                k, atom1, cart1, spin if self.spin else None))
            reader2: ELSIReader = (reader1 if (atom1, cart1) == (atom2, cart2) else
                    ELSIReader(self.get_csc_file_name(
                            k, atom2, cart2, spin if self.spin else None)))
            if domain == 'fourier':
                partial_kernel += spin_weight * kw * self.compute_fourier_term(
                    reader1, reader2, energies, occ, beta, grid, freq_broadening, mode=mode, add_diag=add_diag)
            elif domain == 'laplace':
                raise NotImplementedError
                # TODO: this has to be updated
                # partial_kernel += spin_weight * kw * self.compute_laplace_term(
                #     reader1, reader2, energies, occ, beta, grid, freq_broadening, mode=mode)
            else:
                raise NotImplementedError
                # TODO: this has to be updated
                # partial_kernel += spin_weight * kw * self.compute_temporal_term(
                #     reader1, reader2, energies, occ, beta, grid, freq_broadening, 
                #     excitation_cutoff, mode=mode)
            del reader1
            del reader2
        # Reduce the partial sums to the root process
        if self.comm:
            kernel = np.zeros(partial_kernel.shape, dtype=complex)
            self.comm.Reduce(partial_kernel, kernel, op=MPI.SUM, root=0)
        else:
            kernel = partial_kernel
        if self.root:
            # Mass weighting
            m1: float = self.masses[atom1-1]
            m2: float = self.masses[atom2-1]
            kernel /= np.sqrt(m1*m2)
        # Units
        if domain == 'time':
            grid /= self.ps 
            kernel *= self.ps**2
        else:
            grid *= self.hartree
            kernel *= self.ps
        return grid, kernel
                
    def compute_fourier_term(
            self,
            reader1: ELSIReader,
            reader2: ELSIReader,
            energies: np.ndarray,
            occ: np.ndarray,
            beta: float,
            freqs: np.ndarray,
            freq_broadening: float,
            mode: Optional[int] = 2,
            add_diag: Optional[bool] = False
        ):
        """Calculate the contribution to the Fourier transform of the friction kernel
        from a particular spin channel and k-point, as determined by the files parsed
        by reader1 and reader2

        Args:
            * reader1, reader2 (ELSIReader): see above
            * energies (np.ndarray): Kohn-Sham energies of the orbitals at the current spin and lattice wave vector 
            * occ (np.ndarray): occupation numbers of the orbitals
            * beta (float): 1/kB*T in atomic units
            * freqs (np.ndarray): frequency grid, on which to compute the Fourier transform
            * freq_broadening (float): standard deviation (width) of the Gaussians used as nascent Deltas in the calculation of spectral densities.
            * mode (int, optional): `0` corresponds to original FHI-aims implementation, `1` is the first fix, which extends the lower portion of the energy grid a few broadening widths below zero, and `2` is the final fix, which uses an analytical expression to include de-excitations.
            * add_diag (bool, optional): include diagonal coupling terms from the calculation of the friction (debugging only, DON'T DO THIS for production calculations); default is False.
        """

        freq_step = (freqs[-1]-freqs[0])/(len(freqs)-1)
        Lambda = np.zeros_like(freqs, dtype=complex)
        lo_state, lumo_state, homo_state, hi_state = self._get_energy_manifold_indices(
            energies, occ, freqs[-1])
        for i_col, (Em, fm) in enumerate(zip(energies, occ)):
            if i_col < lo_state or i_col > homo_state:
                continue
            pop_diag = False
            if mode == 0:
                col_slice = slice(max(i_col+1,lumo_state),hi_state,None)
            elif mode == 1:
                if i_col < lumo_state or add_diag:
                    col_slice = slice(lumo_state,hi_state,None)
                elif i_col > lumo_state:
                    col_slice = np.r_[lumo_state:i_col, i_col+1:hi_state]
                else:
                    col_slice = np.r_[i_col+1:hi_state]
            elif mode == 2:
                if add_diag:
                    pop_diag = i_col >= lumo_state
                    if pop_diag:
                        # current state is within the upper manifold
                        col_slice = slice(i_col,hi_state,None)
                    else:
                        col_slice = slice(lumo_state,hi_state,None)
                else:
                    col_slice = slice(max(i_col+1,lumo_state),hi_state,None)
            else:
                raise RuntimeError
            g2 = reader2.read_matrix_column(i_col)[col_slice]
            g1 = np.conj(g2) if reader1 is reader2 else np.conj(
                reader1.read_matrix_column(i_col)[col_slice])
            f = occ[col_slice]
            fdiff = fm - f
            Omega = energies[col_slice] - Em
            if mode == 0:
                ekeep = np.logical_and(
                    Omega > freqs[0], np.logical_and(
                        Omega < freqs[-1], np.abs(fdiff) > 0.0001))
            elif mode == 1:
                ekeep = np.logical_and(
                    Omega > freqs[0] - self.nsigma*freq_broadening,
                    Omega < freqs[-1] + self.nsigma*freq_broadening)
            elif mode == 2:
                ekeep = Omega < freqs[-1] + self.nsigma*freq_broadening
            deltas = self.gaussian_delta(
                freqs[:,None], Omega, freq_broadening, nsigma=self.nsigma)[0]
            if mode == 0:
                norm = np.sum(deltas, axis=0) * freq_step
            else:
                norm = np.ones_like(deltas[0])
            g1g2f = g1*g2*self._fdiff_by_omega(beta, f, fdiff, Omega)
            for i_freq, delta in enumerate(deltas):
                Lambda[i_freq] += np.sum(g1g2f[ekeep] * delta[ekeep] / norm[ekeep])
            if mode == 2:
                if pop_diag:
                    # pop the diagonal term
                    Omega = Omega[1:]
                    g1g2f = g1g2f[1:]
                    ekeep = ekeep[1:]
                # contributions from -ve excitation energies
                deltas = self.gaussian_delta(
                    freqs[:,None], -Omega, freq_broadening, nsigma=self.nsigma)[0]
                g1g2f = np.conj(g1g2f)
                for i_freq, delta in enumerate(deltas):
                    Lambda[i_freq] += np.sum(g1g2f[ekeep] * delta[ekeep])
        return np.pi*Lambda
            
    def compute_laplace_term(
            self,
            reader1: ELSIReader,
            reader2: ELSIReader,
            energies: np.ndarray,
            occ: np.ndarray,
            beta: float,
            freqs: np.ndarray,
            freq_broadening: float,
            debug: Optional[bool] = False):
        """Calculate the contribution to the Laplace transform of the friction kernel
        from a particular spin channel and k-point, as determined by the files parsed
        by reader1 and reader2

        Args:
            reader1, reader2 (ELSIReader): see above
            energies (np.ndarray): Kohn-Sham energies of the orbitals at the current spin and lattice wave vector 
            occ (np.ndarray): occupation numbers of the orbitals
            beta (float): 1/kB*T in atomic units
            freqs (np.ndarray): frequency grid, on which to compute the Fourier transform
            freq_broadening (float): standard deviation (width) of the Gaussians used as nascent Deltas in the calculation of spectral densities.
            debug (Optional[bool], optional): Use the FHI-aims implementation which restricts excitation energies to be strictly positive renormalises the nascent deltas. Defaults to False.
        """
        raise NotImplementedError
        # TODO: this has to be updated and tested
        Lambda = np.zeros_like(freqs, dtype=complex)
        lo_state, lumo_state, homo_state, hi_state = self._get_energy_manifold_indices(
            energies, occ, freqs[-1])
        for i_row, (Em, fm) in enumerate(zip(energies, occ)):
            if i_row < lo_state or i_row > homo_state:
                continue
            if debug:
                hi_slc = slice(max(i_row+1,lumo_state),hi_state,None)
            else:
                hi_slc = slice(lumo_state,hi_state,None)
            g2 = reader2.read_matrix_column(i_row)[hi_slc]
            g1 = np.conj(g2) if reader1 is reader2 else np.conj(
                reader1.read_matrix_column(i_row)[hi_slc])
            f = occ[hi_slc]
            fdiff = fm - f
            Omega = energies[hi_slc] - Em
            lorentz = freqs[:,None] / (freqs[:,None]**2 + Omega**2)
            g1g2f = g1*g2*self._fdiff_by_omega(beta, f, fdiff, Omega)
            steps, keep_lst = self.erf_step(
                Omega, freq_broadening, nsigma=self.nsigma
            )
            for i_freq, (keep, step) in enumerate(zip(keep_lst, steps)):
                Lambda[i_freq] += np.sum(g1g2f[keep] * lorentz[keep] * step[keep])
        return 2*Lambda
    
    def compute_temporal_term(
            self,
            reader1: ELSIReader,
            reader2: ELSIReader,
            energies: np.ndarray,
            occ: np.ndarray,
            beta: float,
            time: np.ndarray,
            freq_broadening: float,
            excitation_cutoff: float,
            debug: Optional[bool] = False):
        """Calculate the contribution to the friction kernel from a particular spin channel
        and k-point, as determined by the files parsed by reader1 and reader2

        Note:
            All values should be in atomic units.

        Args:
            reader1, reader2 (ELSIReader): see above
            energies (np.ndarray): Kohn-Sham energies of the orbitals at the current spin and lattice wave vector 
            occ (np.ndarray): occupation numbers of the orbitals
            beta (float): 1/kB*T in atomic units
            time (np.ndarray): time grid, on which to compute the memory-friction kernel
            freq_broadening (float): standard deviation (width) of the Gaussians used as nascent Deltas in the calculation of spectral densities.
            excitation_cutoff (float): highest excitation energy included in the cosine series expansion of the memory-friction kernel.
            debug (Optional[bool], optional): Use the FHI-aims implementation which restricts excitation energies to be strictly positive renormalises the nascent deltas. Defaults to False.
        """
        raise NotImplementedError
        # TODO: this has to be updated and tested
        eta = np.zeros_like(time, dtype=complex)
        lo_state, lumo_state, homo_state, hi_state = self._get_energy_manifold_indices(
            energies, occ, excitation_cutoff)
        for i_row, (Em, fm) in enumerate(zip(energies, occ)):
            if i_row < lo_state or i_row > homo_state:
                continue
            if debug:
                hi_slc = slice(max(i_row+1,lumo_state),hi_state,None)
            else:
                hi_slc = slice(lumo_state,hi_state,None)
            g2 = reader2.read_matrix_column(i_row)[hi_slc]
            g1 = np.conj(g2) if reader1 is reader2 else np.conj(
                reader1.read_matrix_column(i_row)[hi_slc])
            f = occ[hi_slc]
            fdiff = fm - f
            Omega = energies[hi_slc] - Em
            if debug:
                ekeep = Omega < excitation_cutoff
                cos = np.cos(Omega[ekeep]*time[:,None])
                g1g2f = (g1[ekeep] * g2[ekeep] * self._fdiff_by_omega(
                        beta, f[ekeep], fdiff[ekeep], Omega[ekeep]))
                eta += np.sum(g1g2f * cos, axis=-1)
            else:
                ekeep = np.logical_and(
                    Omega > -self.nsigma*freq_broadening, 
                    Omega < excitation_cutoff + self.nsigma*freq_broadening)
                g1g2f = (g1[ekeep] * g2[ekeep] * self._fdiff_by_omega(
                        beta, f[ekeep], fdiff[ekeep], Omega[ekeep]))
                cos = np.cos(Omega[ekeep]*time[:,None]) 
                step, _ = self.erf_step(
                    Omega[ekeep], freq_broadening, 
                    nsigma=self.nsigma, 
                    upper=excitation_cutoff) # DEBUGGING
                eta += np.sum(g1g2f * cos * step, axis=-1)
        # account for the effect of replacing delta functions by Gaussians in the 
        # frequency domain:
        smear = np.exp(-(time*freq_broadening)**2 / 2)
        eta *= smear
        return 2*eta
    
    def compute_tensor_from_grid(self) -> tuple[np.ndarray, np.ndarray] :
        freqs, Lambda, _ = self.parse_friction_tensor_grid()
        # convert energies to atomic units
        freqs /= self.hartree 
        # time in atomic units
        times = kernel_transform.timefreqs(freqs)
        print(f"{Lambda.shape = }")
        # currently in 1 / ps*t0, where t0 is the atomic unit of time
        eta = kernel_transform.lambda_to_kernel(Lambda, freqs)
        # time in ps, eta in 1/ps^2
        return times / self.ps, eta * self.ps


def main():
    import argparse
    import logging

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
    parser.add_argument('-p', '--spin-polarized',
                        action='store_true',
                        help='Use when the calculation was run with `spin collinear`.')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        choices=[0,1,2],
                        help="Verbosity level, from silent to debug.")
    parser.add_argument('-t', '--tensor', action="store_true",
                        help="Read the static friction tensor.")
    parser.add_argument('-g', '--tensor-grid', type=str, default=None,
                        help="Read the frequency-dependent friction tensor grid and store in a csv file.")
    parser.add_argument('-k', '--kernel', type=str, default=None,
                        help="Compute the mass-weighted memory-friction kernel and store in a csv file.")
    group = parser.add_argument_group("raw", "processing raw coupling elements")
    group.add_argument("-T", type=float, default=300.0,
                       help="Electronic temperature in Kelvin.")
    group.add_argument("--a1", type=int, default=1, 
                       help="Index of the first of two coupled atoms")
    group.add_argument("--c1", type=int, default=1, 
                       help="Cartesian component of the first of two coupled atoms")
    group.add_argument("--a2", type=int, default=1, 
                       help="Index of the second of two coupled atoms")
    group.add_argument("--c2", type=int, default=1, 
                       help="Cartesian component of the second of two coupled atoms")
    group.add_argument("--raw-lambda", action="store_true",
                       help="Calculate the cosine transform of the friction kernel from raw coupling data.")
    # group.add_argument("--raw-kernel", action="store_true",
    #                    help="Calculate the friction kernel from raw coupling data.")
    # group.add_argument("--max-time", type=float, default=0.1,
    #                    help="Maximum time (in ps) for which to calculate the kernel.")
    # group.add_argument("--time-step", type=float, default=None,
    #                    help="Time step (in ps) for the grid on which to "
    #                    "calculate the kernel. By default create a grid of 100 points")
    group.add_argument("--max-freq", type=float, default=2.0,
                       help="Maximum energy (in eV) for which to calculate the kernel in the frequency domain.")
    group.add_argument("--freq-step", type=float, default=None,
                       help="Frequency step (in eV) for the grid on which to "
                       "calculate the kernel. By default create a grid of 100 points")
    group.add_argument("--broadening", type=float, default=0.01, 
                       help="Broadening in the frequency domain, in eV, which corresponds "
                       "to applying a Gaussian window to the kernel in the time domain.")
    group.add_argument("--mode", type=int, choices=[0, 1, 2],
                       help="Calculation mode; `0` corresponds to original FHI-aims implementation, `1` is the first fix, which extends the lower portion of the energy grid a few broadening widths below zero, and `2` is the final fix, which uses an analytical expression to include de-excitations.")
    group.add_argument("--diag", action='store_true', 
                       help="Include the self-interaction contributions from the sum over coupling matrix elements (only has an effect if '--mode' is 1 or 2.")
    
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
    
    friction_parser: FrictionParser = FrictionParser(
        args.dir, args.output, spin_collinear=args.spin_polarized,
        comm=comm, root=root, size=size, rank=rank)
    
    logger.debug(
        f"{friction_parser.aimsdir = }"
    )
    logger.debug(
        f"{friction_parser.aimsout = }\n"
    )
    np.set_printoptions(formatter=dict(float_kind=lambda x: f"{x: .6f}"))
    logger.info(
        "\nK-grid and weights:"
    )
    logger.info(
        np.c_[friction_parser.kgrid, friction_parser.kweights[:,None]]
    )

    logger.info(f"\nFermi level = {friction_parser.e_fermi:.6f} eV")
    logger.debug("\nAtomic masses:")
    logger.debug(friction_parser.masses)
    logger.debug("\nKohn-Sham eigenvalues:")
    if args.spin_polarized:
        logger.debug("Spin-up")
        logger.debug(friction_parser.eigvals[0])
        logger.debug("\nSpin-down")
        logger.debug(friction_parser.eigvals[1])
    else:
        logger.debug(friction_parser.eigvals[0])

    if args.tensor and root:
        print("\nStatic friction tensor [1/ps]:")
        tensor = friction_parser.parse_friction_tensor()
        for row in tensor:
            print('['+','.join([f'{x: .3f}' for x in row])+']')

    if args.kernel and root:
        print("Computing the memory-friction kernel...")
        time, eta = friction_parser.compute_tensor_from_grid()
        print("Done")
        print(f"Saving data to file '{args.kernel}'")
        dim = len(eta)
        ncoord = int(round(np.sqrt(2*dim+0.25)-0.5))
        triu = np.triu_indices(ncoord)
        R, C = np.indices(2*(ncoord,))
        r, c = R[triu], C[triu]
        indices = [(x+1, y+1) for x, y in zip(r, c)]
        data = {"time [ps]" : time}
        for idx, col in zip(indices, eta):
            data.update({idx : col})
        df = pd.DataFrame(data=data)
        df.to_csv(
            args.kernel,
            float_format="%.6f",
            index=False
        )

    if (args.tensor_grid is not None) and root:
        print("Parsing the frequency-dependent memory-friction spectral density...")
        freqs, Lambda, _ = friction_parser.parse_friction_tensor_grid()
        print("Done")
        print(f"Saving data to file '{args.tensor_grid}'")
        dim = len(Lambda)
        ncoord = int(round(np.sqrt(2*dim+0.25)-0.5))
        triu = np.triu_indices(ncoord)
        R, C = np.indices(2*(ncoord,))
        r, c = R[triu], C[triu]
        indices = [(x+1, y+1) for x, y in zip(r, c)]
        data = {"energy [eV]" : freqs}
        for idx, col in zip(indices, Lambda):
            data.update({idx : col})
        df = pd.DataFrame(data=data)
        df.to_csv(
            args.tensor_grid,
            float_format="%.6f",
            index=False
        )

    # if args.raw_kernel:
    #     logger.info("Calculating memory-friction kernel from raw "
    #                 "coupling elements\nfor the (atom, cart) pair "
    #                 f"{args.a1, args.c1}--{args.a2, args.c2} "
    #                 f"at a temperature of {args.T:.2f} K")
    #     time_step = args.time_step if args.time_step else args.max_time / 99
    #     times, eta = friction_parser.compute_tensor_element(
    #         args.a1, args.c1, args.a2, args.c2, 
    #         args.max_time, time_step, domain='time',
    #         T=args.T, freq_broadening=args.broadening,
    #         excitation_cutoff=args.max_freq, mode=args.mode)
    #     output = (f"eta_atom_{args.a1:06d}_cart_{args.c1:1d}"
    #               f"_{args.a2:06d}_cart_{args.c2:1d}"
    #               f"_temp_{args.T:.2f}K.csv")
    #     logging.info(f"Saving kernel to {output}.")
    #     if root:
    #         data = {"time [ps]" : times, "kernel [1/ps²]" : eta.real}
    #         df = pd.DataFrame(data=data)
    #         df.to_csv(
    #             output,
    #             float_format="%.6f",
    #             index=False
    #         )

    if args.raw_lambda:
        logger.info("Calculating cosine transform of memory-friction kernel from raw "
                    "coupling elements\nfor the (atom, cart) pair "
                    f"{args.a1, args.c1}--{args.a2, args.c2} "
                    f"at a temperature of {args.T:.2f} K")
        freq_step = args.freq_step if args.freq_step else args.max_freq / 99
        freqs, Lambda = friction_parser.compute_tensor_element(
            args.a1, args.c1, args.a2, args.c2, 
            args.max_freq, freq_step, domain='fourier',
            T=args.T, freq_broadening=args.broadening, mode=args.mode, add_diag=args.diag)
        output = (f"Lambda_atom_{args.a1:06d}_cart_{args.c1:1d}"
                  f"_{args.a2:06d}_cart_{args.c2:1d}"
                  f"_temp_{args.T:.2f}K.csv")
        logging.info(f"Saving Lambda to {output}.")
        if root:
            data = {"freqs [eV]" : freqs, "Lambda [1/ps]" : Lambda.real}
            df = pd.DataFrame(data=data)
            df.to_csv(
                output,
                float_format="%.6f",
                index=False
            )
            
if __name__ == "__main__":
        main()
