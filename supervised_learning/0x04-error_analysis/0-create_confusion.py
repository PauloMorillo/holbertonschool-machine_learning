#!/usr/bin/env python3
"""This module has the method create_confusion_matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """This method create a confusion matrix"""
    return(np.matmul(labels.T, logits))
