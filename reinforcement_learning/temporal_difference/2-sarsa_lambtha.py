#!/usr/bin/env python3
"""
SARSA(λ) algorithm with eligibility traces for RL
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects the next action using epsilon-greedy policy.

    Parameters:
        Q (numpy.ndarray): Q-table with shape (s, a)
        state (int): Current state
        epsilon (float): Epsilon for epsilon-greedy policy

    Returns:
        int: Selected action
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(0, Q.shape[1])  # Explore
    else:
        return np.argmax(Q[state])  # Exploit


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) algorithm to estimate Q-table values.

    Parameters:
        env (gym.Env): Environment instance
        Q (numpy.ndarray): Q-table of shape (s, a)
        lambtha (float): Eligibility trace factor
        episodes (int): Number of episodes
        max_steps (int): Max steps per episode
        alpha (float): Learning rate
        gamma (float): Discount rate
        epsilon (float): Initial epsilon for epsilon-greedy policy
        min_epsilon (float): Minimum epsilon value
        epsilon_decay (float): Decay rate for epsilon

    Returns:
        numpy.ndarray: Updated Q-table
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]  # Reset environment and get initial state
        action = epsilon_greedy(Q, state, epsilon)  # Choose initial action

        # Initialize eligibility traces
        eligibility_traces = np.zeros_like(Q)

        for step in range(max_steps):
            # Take action and observe new state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose next action based on epsilon-greedy policy
            next_action = epsilon_greedy(Q, next_state, epsilon)

            # Calculate TD error
            delta = reward + gamma * Q[next_state, next_action] - Q[state,
                                                                    action]

            # Update eligibility trace for the current state-action pair
            eligibility_traces[state, action] += 1

            # Update Q values and decay eligibility traces
            Q += alpha * delta * eligibility_traces
            eligibility_traces *= gamma * lambtha  # Decay eligibility traces

            # Move to the next state and action
            state = next_state
            action = next_action

            if terminated or truncated:
                break

        # Decay epsilon after each episode
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))

    return Q
