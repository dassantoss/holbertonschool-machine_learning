#!/usr/bin/env python3
"""
TD(λ) algorithm for estimating the value function in RL
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha=0.9, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm for estimating the value function.

    Parameters:
        env: Environment instance.
        V (numpy.ndarray): Value estimates of shape (s,).
        policy: Function that takes a state and returns the next action.
        lambtha (float): The eligibility trace decay parameter.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount rate.

    Returns:
        numpy.ndarray: Updated value estimates V.
    """
    for episode in range(episodes):
        # Reset the environment and get the initial state
        state = env.reset()[0]

        # Initialize eligibility traces to zero for all states
        eligibility_traces = np.zeros_like(V)

        for step in range(max_steps):
            # Select action based on policy
            action = policy(state)

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Calculate the TD error (delta)
            delta = reward + (gamma * V[next_state]) - V[state]

            # Update eligibility trace for the current state
            eligibility_traces[state] += 1

            # Update all states' value estimates and eligibility traces
            V += alpha * delta * eligibility_traces

            # Apply decay to the eligibility traces
            eligibility_traces *= gamma * lambtha

            # Move to the next state
            state = next_state

            if terminated or truncated:
                break

    return V
