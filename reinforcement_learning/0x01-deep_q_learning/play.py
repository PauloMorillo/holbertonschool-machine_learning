#!/usr/bin/env python3
"""
This module play the game according to the train before
"""

import gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
import keras as K

create_q_model = __import__('train').create_q_model
AtariProcessor = __import__('train').AtariProcessor

if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    l_window = 4
    model = create_q_model(num_actions, l_window)
    mem = SequentialMemory(limit=1000000, window_length=l_window)
    processor = AtariProcessor()
    dqn = DQNAgent(model=model,
                   nb_actions=num_actions,
                   processor=processor,
                   memory=mem)
    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=10, visualize=True)
