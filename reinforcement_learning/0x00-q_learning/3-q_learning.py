#!/usr/bin/env python3
"""
This module has the method
def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
"""
import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    This method  performs Q-learning
    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, total_rewards
    """
    total_r = []
    f_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()
        sum_r = 0
        for t in range(max_steps):
            # env.render()
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -1

            # Update Q-table for Q(s, a)
            old_value = Q[state, action]
            next_max = np.max(Q[new_state])

            new_value = (1 - alpha) * old_value + \
                        (alpha * (reward + gamma * next_max))
            Q[state, action] = new_value
            sum_r += reward
            state = new_state

            if done:
                break
        # Exploration rate decay
        epsilon = min_epsilon + (f_epsilon - min_epsilon) *\
            np.exp(-epsilon_decay * episode)
        total_r.append(sum_r)

    return Q, total_r
