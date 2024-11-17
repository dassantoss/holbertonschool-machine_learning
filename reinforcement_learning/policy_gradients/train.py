#!/usr/bin/env python3
"""
Training Loop for Monte-Carlo Policy Gradient with Rendering
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Train the policy using Monte-Carlo policy gradient.

    Parameters:
        env: initial environment
        nb_episodes: number of episodes used for training
        alpha: the learning rate
        gamma: the discount factor
        show_result: if True, renders the environment every 1000 episodes

    Returns:
        list: Score values (rewards obtained during each episode)
    """
    # Initialize weights
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for episode in range(nb_episodes):
        # Reset the environment and get initial state
        state = env.reset()[0]
        episode_rewards = []
        episode_gradients = []
        done = False

        while not done:
            # Get action and gradient
            action, gradient = policy_gradient(state, weights)

            # Render the environment every 1000 episodes show_result is True
            if show_result and episode % 1000 == 0:
                env.render()

            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Record rewards and gradients
            episode_rewards.append(reward)
            episode_gradients.append(gradient)

            state = next_state
            done = terminated or truncated

        # Compute total score for the episode
        score = sum(episode_rewards)
        scores.append(score)

        # Update weights based on returns and gradients
        for i, gradient in enumerate(episode_gradients):
            # Cumulative discounted rewards
            reward = \
                sum([R * gamma ** idx for idx,
                     R in enumerate(episode_rewards[i:])])
            weights += alpha * gradient * reward

        # Print score for debugging
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {score}")

    return scores
