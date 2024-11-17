#!/usr/bin/env python3
"""Module to compute Monte-Carlo policy gradient."""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy (probabilities of actions) using softmax.

    Args:
        matrix (np.ndarray): The state matrix of shape (1, n).
        weight (np.ndarray): The weight matrix of shape (n, m).

    Returns:
        np.ndarray: Probabilities of actions of shape (1, m).
    """
    z = np.dot(matrix, weight)
    exp_z = np.exp(z - np.max(z))  # Stability improvement
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient.

    Args:
        state (np.ndarray): Current state of the environment (shape (n,)).
        weight (np.ndarray): Weight matrix (shape (n, m)).

    Returns:
        tuple: (action, gradient) where:
               - action: int, selected action.
               - gradient: np.ndarray, gradient of shape (n, m).
    """
    # Reshape state for matrix multiplication
    state = state.reshape(1, -1)

    # Get action probabilities
    probs = policy(state, weight)

    # Sample an action
    action = np.random.choice(probs.shape[1], p=probs.flatten())

    # One-hot encode the selected action
    action_one_hot = np.zeros_like(probs)
    action_one_hot[0, action] = 1

    # Compute the gradient
    gradient = np.dot(state.T, (action_one_hot - probs))

    return action, gradient
