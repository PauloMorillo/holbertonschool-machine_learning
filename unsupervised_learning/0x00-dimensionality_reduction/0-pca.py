#!/usr/bin/env python3
""" This module has the method pca(X, var=0.95)"""

import numpy as np


def pca(X, var=0.95):
    """ Ths method perfoms PCA on a dataset"""
    u, s, v = np.linalg.svd(X)
    pos = np.where((np.cumsum(s) / np.sum(s)) >= var)[0][0]
    v_filtered = v[:pos + 1]
    return v_filtered.T
