#!/usr/bin/env python3
"""This module has the method specificity"""
import numpy as np


def specificity(confusion):
    """This method calculates the specificity
    for each class in a confusion matrix"""
    allv = confusion.T.sum(axis=1)
    allh = confusion.sum(axis=1)
    tp = confusion.diagonal()
    true_negative = tp
    fp = allv - tp
    print(fp)
    tn = tp - (allh + allv)
    return tn / (tn + fp)
