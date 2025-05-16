#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   mympi.py
@Time    :   2024/08/07 16:03:47
@Author  :   George Trenins
@Desc    :   Define a dummy MPI class for running serial calculations when the MPI environment is not set up
'''


from __future__ import print_function, division, absolute_import


class Intracomm(object):
    """A dummy communicator class."""
    
    pass