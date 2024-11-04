#!/usr/bin/env python3
"""
DQN training script for Breakout using Keras-RL.
"""
import gymnasium as gym
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor


class AtariProcessor(Processor):
    """
    Processor to preprocess observations for training.
    Converts observations to grayscale, resizes, and normalizes.
    """

    def process_observation(self, observation):
        """Convert observation to grayscale and resize to 84x84."""
        if isinstance(observation, tuple):
            observation = observation[0]
        observation = cv2.cvtColor(np.array(observation), cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (84, 84))
        return observation

    def process_state_batch(self, batch):
        """Normalize batch values to float32 and scale to [0, 1]."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip reward to be within [-1, 1] range."""
        return np.clip(reward, -1., 1.)


class CompatibilityWrapper(gym.Wrapper):
    """
    Wrapper to ensure compatibility with older Gym versions.
    Adjusts step and reset methods to use `done` instead of
    `terminated` and `truncated`.
    """

    def step(self, action):
        """Executes action in environment and handles done flag."""
        observation, reward, terminated, truncated, info = \
            self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        """Resets environment and returns initial observation."""
        observation, info = self.env.reset(**kwargs)
        return observation


def build_model(input_shape, nb_actions):
    """
    Builds a convolutional neural network model.

    Args:
        input_shape (tuple): Shape of input observations.
        nb_actions (int): Number of possible actions.

    Returns:
        keras.models.Sequential: Compiled CNN model.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def build_agent(model, nb_actions):
    """
    Constructs the DQN agent with memory and policy settings.

    Args:
        model (keras.models.Sequential): Model representing Q-network.
        nb_actions (int): Number of possible actions in environment.

    Returns:
        rl.agents.DQNAgent: Configured DQN agent ready for training.
    """
    memory = SequentialMemory(limit=100000, window_length=4)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=50000
    )
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=5000,
                   target_model_update=1e-2,
                   processor=AtariProcessor(),
                   gamma=0.99,
                   policy=policy,
                   train_interval=4)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


def main():
    """
    Main function to create environment, build model, and start training.
    Saves trained weights after completion.
    """
    env = gym.make("ALE/Breakout-v5")
    env = CompatibilityWrapper(env)
    nb_actions = env.action_space.n
    model = build_model((4, 84, 84), nb_actions)
    dqn = build_agent(model, nb_actions)

    dqn.fit(env, nb_steps=200000, visualize=False, verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)


if __name__ == "__main__":
    main()
