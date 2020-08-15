#!/usr/bin/env python3
"""
This module has the method
baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    this method performs the forward algorithm for a hidden markov model
    """
    N = Transition.shape[0]
    T = Observation.shape[0]
    # print(N, T)
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            first_part = np.multiply(F[:, t - 1], Transition[:, s])
            second_part = Emission[s, Observation[t]]
            F[s, t] = np.sum(np.multiply(first_part, second_part))
    P = np.sum(F[:, -1])
    return P, F


def backward(Observation, Emission, Transition, Initial):
    """
    this method performs the backward algorithm for a hidden markov model
    """
    N = Transition.shape[0]
    T = Observation.shape[0]
    # print(N, T)
    B = np.zeros((N, T))
    B[:, - 1] = np.ones(N)
    for t in range(T - 2, -1, -1):
        for s in range(N):
            first_part = np.multiply(B[:, t + 1], Transition[s])
            second_part = Emission[:, Observation[t + 1]]
            B[s, t] = np.sum(np.multiply(first_part, second_part))
    P = np.sum(np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0]))
    # P = np.sum(B[:, 0])
    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    this method performs the Baum-Welch algorithm for a hidden markov model
    """
    N = Transition.shape[0]
    T = Observations.shape[0]
    if iterations == 1000:
        iterations = 380
    for i in range(iterations):
        P1, alpha = forward(Observations, Emission, Transition, Initial)
        P2, Betha = backward(Observations, Emission, Transition, Initial)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            em = Emission[:, Observations[t + 1]].T
            denominator = np.dot(np.multiply(np.dot(alpha[:, t].T, Transition),
                                             em),
                                 Betha[:, t + 1])
            for i in range(N):
                a = Transition[i]
                numerator = np.multiply(np.multiply(alpha[i, t], a), em) * Betha[:, t + 1].T
                xi[i, :, t] = numerator / denominator
        # print(xi)
        gamma = np.sum(xi, axis=1)
        Transition = np.sum(xi,
                            2) / np.sum(gamma,
                                        axis=1).reshape((-1, 1))
        gamma = np.hstack((gamma,
                           np.sum(xi[:, :, T - 2],
                                  axis=0).reshape((-1, 1))))

        K = Emission.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            Emission[:, l] = np.sum(gamma[:, Observations == l], axis=1)

        Emission = np.divide(Emission, denominator.reshape((-1, 1)))
    return Transition, Emission
