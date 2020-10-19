#!/usr/bin/env python3
"""
This module train a model to play atari
"""

import gym
import keras as K
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from keras import layers
from rl.processors import Processor
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from PIL import Image
import numpy as np

import tensorflow as tf

tf.compat.v1.disable_eager_execution()


class AtariProcessor(Processor):
    """
    This class instace an Atari Processor
    """

    def process_observation(self, observation):
        """
        This method preprocessing
        """
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize((84, 84), Image.ANTIALIAS).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        This module normalize the batch
        """
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """ This method sets the rewards"""
        return np.clip(reward, -1., 1.)


def create_q_model(num_actions, window):
    """
     This method create keras model
    """

    inputs = layers.Input(shape=(window, 84, 84))
    inputs_sort = layers.Permute((2, 3, 1))(inputs)
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu",
                           data_format="channels_last")(inputs_sort)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu",
                           data_format="channels_last")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu",
                           data_format="channels_last")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)
    return K.Model(inputs=inputs, outputs=action)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_actions = env.action_space.n
    l_window = 4
    model = create_q_model(num_actions, l_window)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=l_window)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.,
                                  value_min=.1,
                                  value_test=.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model,
                   nb_actions=num_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)
    dqn.compile(K.optimizers.Adam(lr=.00025), metrics=['mae'])
    dqn.fit(env,
            nb_steps=17500,
            log_interval=10000,
            visualize=False,
            verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
