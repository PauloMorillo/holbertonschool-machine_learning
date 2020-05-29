#!/usr/bin/env python3
"""This module has the method early_stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """This method determines if you should stop gradient descent early"""
    delta = opt_cost - cost
    verify = False
    if delta > threshold:
        count = 0
    else:
        count = count + 1
    if patience == count:
        verify = True
    return verify, count
