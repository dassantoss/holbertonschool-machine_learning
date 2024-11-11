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
        state = env.reset()[0]  # Reset environment and get initial state
        episode_data = []  # Store state-reward pairs for this episode

        for step in range(max_steps):
            action = policy(state)  # Get action from policy
            next_state, reward, done, _, _ = env.step(action)  # Take action
            episode_data.append((state, reward))  # Save state and reward
            state = next_state  # Move to next state
            if done:
                break  # End episode if we reach a terminal state

        # Calculate the return G for each state in the episode
        G = 0  # Initialize the return
        visited_states = set()

        # Traverse episode_data in reverse order for G calculation
        for state, reward in reversed(episode_data):
            G = reward + gamma * G

            # Update V only if the state has not been visited in this episode
            if state not in visited_states:
                visited_states.add(state)
                # Incremental update of the state value
                V[state] = V[state] + alpha * (G - V[state])

    return V
