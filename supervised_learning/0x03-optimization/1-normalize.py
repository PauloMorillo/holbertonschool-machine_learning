#!/usr/bin/env python3
""" This module has the method normalize"""
import numpy as np


def normalize(X, m, s):
    """
    This method normalizes (standardizes) a matrix
    """
    X = np.divide(X - m, s)
    return X
