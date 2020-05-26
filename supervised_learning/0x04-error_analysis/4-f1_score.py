#!/usr/bin/env python3
"""This module has the method f1_score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """This method calculates the f1 score
    of a confusion matrix"""
    numerator = np.multiply(precision(confusion), sensitivity(confusion))
    denominator = precision(confusion) + sensitivity(confusion)
    return (2 * numerator) / denominator
