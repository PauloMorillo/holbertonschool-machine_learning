#!/usr/bin/env python3
"""This module has the method specificity"""
import numpy as np


def specificity(confusion):
    """This method calculates the specificity
    for each class in a confusion matrix"""
    allv = confusion.sum(axis=0)
    allh = confusion.sum(axis=1)
    diagonal = confusion.diagonal()
    n = allh.sum() - allh
    fp = allv - diagonal
    tn2 = n - fp
    tn = diagonal.sum() - diagonal
    # print(tn / (tn + fp))
    return tn2 / n
