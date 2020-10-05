#!/usr/bin/env python3
"""
This module has the method
def epsilon_greedy(Q, state, epsilon):
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    This method uses epsilon-greedy to determine the next action
    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation
    Return: next action index
    """

    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])  # Explore action space
    else:
        action = np.argmax(Q[state])  # Exploit learned values

    return action
