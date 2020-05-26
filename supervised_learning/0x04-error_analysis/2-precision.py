#!/usr/bin/env python3
"""This module has the method precision"""
import numpy as np


def precision(confusion):
    """This method calculates the precision
    for each class in a confusion matrix"""
    allv = confusion.T.sum(axis=1)
    diagonal = confusion.diagonal()
    return(diagonal / allv)
