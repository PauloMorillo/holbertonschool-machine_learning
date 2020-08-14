#!/usr/bin/env python3
"""
This module has the method
backward(Observation, Emission, Transition, Initial)
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    this method performs the backward algorithm for a hidden markov model
    Observation is a numpy.ndarray of shape (T,) that contains the index
    of the observation
    T is the number of observations
    Emission is a numpy.ndarray of shape (N, M) containing the emission
    probability of a specific observation given a hidden state
    Emission[i, j] is the probability of observing j given the hidden state i
    N is the number of hidden states
    M is the number of all possible observations
    Transition is a 2D numpy.ndarray of shape (N, N) containing the transition
    probabilities
    Transition[i, j] is the probability of transitioning from the hidden
    state i to j
    Initial a numpy.ndarray of shape (N, 1) containing the probability of
    starting in a particular hidden state
    Returns: P, B, or None, None on failure
    Pis the likelihood of the observations given the model
    B is a numpy.ndarray of shape (N, T) containing the backward
    path probabilities
    B[i, j] is the probability of generating the future observations from
    hidden state i at time j
    """
    N = Transition.shape[0]
    T = Observation.shape[0]
    # print(N, T)
    B = np.zeros((N, T))
    B[:, - 1] = np.ones(N)
    for t in range(T - 2, -1, -1):
        for s in range(N):
            first_part = B[:, t + 1] * Transition[s]
            second_part = Emission[:, Observation[t + 1]]
            B[s, t] = np.sum(first_part * second_part)
    P = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0]))
    # P = np.sum(B[:, 0])
    return P, B
