#!/usr/bin/env python3
"""
This module has the method
forward(Observation, Emission, Transition, Initial)
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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
    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))
    viterbi[:, 0] = Initial.T * Emission[:, Observation[0]]
    backpointer[:, 0] = 0
    for t in range(1, T):
        for s in range(N):
            first_part = viterbi[:, t - 1] * Transition[:, s]
            second_part = Emission[s, Observation[t]]
            viterbi[s, t] = np.max(first_part * second_part)
            backpointer[s, t] = np.argmax(first_part * second_part)
    P = np.max(viterbi[:, -1])
    S = np.zeros(T)
    last_state = np.argmax(viterbi[:, -1])
    # print(last_state)
    S[0] = last_state
    backtrack_index = 1
    for i in range(T - 1, 0, -1):
        S[backtrack_index] = backpointer[int(last_state), i]
        # print(S)
        # print(backpointer[last_state, i])
        last_state = backpointer[int(last_state), i]
        backtrack_index += 1
    S = np.flip(S)
    # print(S.shape)
    path = np.argmax(viterbi[:, -1])
    return S, P
