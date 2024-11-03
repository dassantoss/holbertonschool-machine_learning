#!/usr/bin/env python3
"""
Module to train an agent using Q-learning on the FrozenLake environment.
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Trains the agent using the Q-learning algorithm.

    Parameters:
    ----------
    env : gym.Env
        The FrozenLake environment instance.
    Q : numpy.ndarray
        The Q-table to be updated.
    episodes : int, optional
        Total number of episodes for training.
    max_steps : int, optional
        Maximum number of steps per episode.
    alpha : float, optional
        Learning rate.
    gamma : float, optional
        Discount rate.
    epsilon : float, optional
        Initial epsilon for epsilon-greedy policy.
    min_epsilon : float, optional
        Minimum value that epsilon should decay to.
    epsilon_decay : float, optional
        Decay rate for updating epsilon between episodes.

    Returns:
    -------
    Q : numpy.ndarray
        Updated Q-table after training.
    total_rewards : list of float
        Rewards per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]  # Reset and get initial state
        episode_reward = 0

        for step in range(max_steps):
            # Select action using epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon)

            # Perform action in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Set reward to -1 if agent falls in a hole
            if done and reward == 0:
                reward = -1

            # Update Q-table using Q-learning update rule
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])

            # Accumulate reward
            episode_reward += reward

            # Update state
            state = next_state

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

        # Store the total reward for this episode
        total_rewards.append(episode_reward)

    return Q, total_rewards
