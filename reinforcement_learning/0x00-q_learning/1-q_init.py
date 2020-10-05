#!/usr/bin/env python3
"""
This module has the method
def q_init(env):
"""

import numpy as np


def q_init(env):
    """
    This method initializes the Q-table
    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
