#!/usr/bin/env python3
"""This module has the method sensitivity"""
import numpy as np


def sensitivity(confusion):
    """This method calculates the sensitivity
    for each class in a confusion matrix"""
    allh = confusion.sum(axis=1)
    diagonal = confusion.diagonal()
    return(diagonal / allh)
