#!/usr/bin/env python3
""" This module has the moving_average method"""
import numpy as np


def moving_average(data, beta):
    """
    This method calculates the weighted
    moving average of a data set
    """
    ans = []
    vbef = 0
    for i in range(len(data)):
        print(data[i])
        v = beta * vbef + (1 - beta) * data[i]
        avg = v / (1 - (beta ** (i + 1)))
        ans = ans + [avg]
        vbef = v
    return ans
