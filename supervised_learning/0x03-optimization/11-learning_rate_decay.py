#!/usr/bin/env python3
""" This module has the learning_rate_decay method"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using inverse time decay
    """
    alpha = (alpha / (1 + decay_rate * np.floor(global_step / decay_step)))
    return alpha
