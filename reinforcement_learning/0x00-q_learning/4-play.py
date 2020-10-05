#!/usr/bin/env python3
"""
This module has the method
def play(env, Q, max_steps=100):
"""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def play(env, Q, max_steps=100):
    """
    This method has the trained agent play an episode
    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode
    """
    state = env.reset()
    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        if done:
            env.render()
            break
        state = new_state
    return reward
