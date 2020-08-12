#!/usr/bin/env python3
"""
This module has the method
forward(Observation, Emission, Transition, Initial)
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    this method performs the forward algorithm for a hidden markov model
    Observation is a numpy.ndarray of shape (T,) that contains the index
    of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden
    state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing
    the transition probabilities
    Transition[i, j] is the probability of transitioning from the hidden
    state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability
    of starting in a particular hidden state
    Returns: P, F, or None, None on failure
    P is the likelihood of the observations given the model
    F is a numpy.ndarray of shape (N, T) containing the forward
    path probabilities
    F[i, j] is the probability of being in hidden state i at time j given
    the previous observations
    """
    N = Transition.shape[0]
    T = Observation.shape[0]
    # print(N, T)
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        first_part = F[:, t - 1].dot(Transition[:, :])
        second_part = Emission[:, Observation[t]]
        F[:, t] = first_part * second_part
    P = np.sum(F[:, -1])
    return P, F
