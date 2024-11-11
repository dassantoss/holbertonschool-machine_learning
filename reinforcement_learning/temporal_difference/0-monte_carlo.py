#!/usr/bin/env python3
"""Monte Carlo algorithm for policy evaluation in RL"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Parameters:
    env: environment instance
    V (numpy.ndarray): Value estimate of shape (s,)
    policy (function): Function that takes a state and returns the next action
    episodes (int): Total number of episodes to train over
    max_steps (int): Maximum steps per episode
    alpha (float): Learning rate
    gamma (float): Discount rate

    Returns:
    numpy.ndarray: Updated value estimate V
    """
    for episode in range(episodes):
        # Generate an episode
        state = env.reset()[0]
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, reward))
            state = next_state
            if done:
                break

        # Calculate the return for each state in the episode
        G = 0  # Return
        visited_states = set()

        for state, reward in reversed(episode_data):
            G = reward + gamma * G  # Accumulated return
            if state not in visited_states:
                visited_states.add(state)
                # Update the value estimate V for state
                V[state] = V[state] + alpha * (G - V[state])

    return V
