#!/usr/bin/env python3
""" This module has the learning_rate_decay method"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay
    """
    alpha = 1 / (1 + decay_rate * int(global_step/decay_step)) * alpha
    return alpha
