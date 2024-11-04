#!/usr/bin/env python3
"""
Script to play a game of Breakout using a pre-trained DQN agent.
"""
import gymnasium as gym
import numpy as np
import cv2
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.processors import Processor


class AtariProcessor(Processor):
    """Processes observations for Breakout gameplay."""

    def process_observation(self, observation):
        """Converts observation to grayscale and resizes to 84x84."""
        if isinstance(observation, tuple):
            observation = observation[0]
        observation = cv2.cvtColor(np.array(observation), cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (84, 84))
        return observation

    def process_state_batch(self, batch):
        """Normalizes batch values to be in [0, 1]."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clips the reward to a range between -1 and 1."""
        return np.clip(reward, -1., 1.)


class CompatibilityWrapper(gym.Wrapper):
    """
    Ensures compatibility with the Keras-RL library for older Gym versions.
    """

    def step(self, action):
        observation, reward, terminated, truncated, info = \
            self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation

    def render(self, *args, **kwargs):
        return self.env.render()  # Direct call to render without arguments


def build_model(input_shape, nb_actions):
    """Builds a CNN model with the same architecture as used in training."""
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def build_agent(model, nb_actions):
    """Configures the DQN agent with GreedyQPolicy for testing."""
    memory = SequentialMemory(limit=100000, window_length=4)
    policy = GreedyQPolicy()  # Uses only exploitation (no exploration)
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   processor=AtariProcessor(),
                   policy=policy,
                   gamma=0.99,
                   train_interval=4)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


def main():
    """Main function to load model, configure agent, and test on Breakout."""
    # Step 1: Initialize environment with compatibility wrapper
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = CompatibilityWrapper(env)

    # Step 2: Get action space and build model
    nb_actions = env.action_space.n
    model = build_model((4, 84, 84), nb_actions)
    model.load_weights('policy.h5')

    # Step 3: Configure agent and load the model
    dqn = build_agent(model, nb_actions)

    # Step 4: Test agent on Breakout
    dqn.test(env, nb_episodes=5, visualize=True)
    print("Gameplay completed.")

    env.close()


if __name__ == "__main__":
    main()
