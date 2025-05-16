#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   read_elsi.py
@Time    :   2024/08/01 11:32:48
@Author  :   George Trenins
@Desc    :   Parse CSC files column by column
'''


from __future__ import print_function, division, absolute_import
from typing import Optional
import struct
import numpy as np
from scipy.sparse import csc_array


class ELSIReader(object):

    i8 = "l"  # C type of 'long'
    i4 = "i"  # C type of 'int'
    r8 = "d"  # C type of 'double'
    header_fmt = i8*16
    header_nbytes = struct.calcsize(header_fmt)

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.read_elsi_header()
        self.unpack_elsi_header()
        self.read_indices_and_pointers()
        return

    def read_elsi_header(self) -> None :
        with open(self.filename, 'rb') as f:
            data = f.read(self.header_nbytes)
            self.header: tuple[int] = struct.unpack(self.header_fmt,data)
        return

    def unpack_elsi_header(self) -> None:
        self.complex = bool(self.header[2])
        self.n_basis = self.header[3]
        self.n_electrons = self.header[4]
        self.nnz = self.header[5]
        self.elem_fmt = (2 if self.complex else 1) * self.r8
        self.elem_nbytes=struct.calcsize(self.elem_fmt)
        return
    
    def read_indices_and_pointers(self) -> None:
        with open(self.filename, 'rb') as f:
            f.seek(self.header_nbytes)
            # Get column pointer
            self.col_ptr_nbytes = self.n_basis * struct.calcsize(self.i8)
            col_ptr_buffer = f.read(self.col_ptr_nbytes)
            self.col_ptr = np.concatenate([
                np.frombuffer(col_ptr_buffer, dtype=np.int_).copy(), [self.nnz+1]])
            self.col_ptr -= 1
            # Get row indices
            self.row_idx_nbytes = self.nnz * struct.calcsize(self.i4)
            row_idx_buffer = f.read(self.row_idx_nbytes)
            self.row_idx = np.frombuffer(row_idx_buffer, dtype=np.int32).copy()
        self.row_idx -= 1
        return

    def read_matrix_column(self, i_col: int) -> np.ndarray:
        with open(self.filename, 'rb') as f:
            # skip header, row/column specs and preceding data
            start, end = self.col_ptr[i_col:i_col+2]
            size = end - start
            f.seek(self.header_nbytes + 
                   self.col_ptr_nbytes + 
                   self.row_idx_nbytes + 
                   start * self.elem_nbytes)
            # read data for requested column
            col_data_nbytes = size*self.elem_nbytes
            column_buffer = f.read(col_data_nbytes)
            col_data = np.frombuffer(column_buffer, dtype=np.float64)
        re = col_data[::2]
        im = col_data[1::2]
        col = np.zeros(self.n_basis, dtype=complex)
        col[self.row_idx[start:end]] = re + 1j*im
        return col


class ELSIWriter(object):

    i8 = "l"  # C type of 'long'
    i4 = "i"  # C type of 'int'
    r8 = "d"  # C type of 'double'
    header_fmt = i8*16
    header_nbytes = struct.calcsize(header_fmt)
    UNSET = -910910
    FILEVERSION = 170915

    @staticmethod
    def write_csc_matrix(
            filename: str,
            csc: csc_array,
            n_electrons: Optional[int] = None) -> None:
        header = ELSIWriter.construct_elsi_header(csc, n_electrons=n_electrons)
        ELSIWriter.write_elsi_header(filename, header)
        ELSIWriter.write_indices_and_pointers(filename, csc)
        data = csc.data.tobytes()
        with open(filename, 'ab') as f:
            f.write(data)
        return
    
    @staticmethod
    def construct_elsi_header(
            csc: csc_array,
            n_electrons: Optional[int] = None) -> tuple[int, ...]:
        header = [ELSIWriter.FILEVERSION, ELSIWriter.UNSET]
        if np.iscomplexobj(csc):
            header.append(1)
        else:
            header.append(0)
        n_basis, tmp = csc.shape
        if n_basis != tmp:
            raise RuntimeError(f"Expected a square matrix, instead got shape({n_basis}, {tmp})")
        header.append(n_basis)
        if n_electrons is None:
            header.append(ELSIWriter.UNSET)
        else:
            header.append(n_electrons)
        header.append(csc.nnz)
        header.extend(10*[ELSIWriter.UNSET])
        return tuple(header)
        
    @staticmethod
    def write_elsi_header(
            filename: str,
            header: tuple[int,...]) -> None:
        data = struct.pack(ELSIWriter.header_fmt, *header)
        with open(filename, 'wb') as f:
            f.write(data)
        return

    @staticmethod
    def write_indices_and_pointers(
            filename: str,
            csc: csc_array) -> None:
        indices: np.ndarray = csc.indices.astype(np.int32, casting="same_kind")
        indices += np.int32(1)
        index_buffer = indices.tobytes()
        pointers: np.ndarray = csc.indptr[:-1].astype(np.int64, casting="same_kind")
        pointers += np.int64(1)
        ptr_buffer = pointers.tobytes()
        with open(filename, 'ab') as f:
            f.write(ptr_buffer)
            f.write(index_buffer)
        return