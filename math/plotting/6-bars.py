#!/usr/bin/env python3
"""
This module provides functionality to plot a stacked bar graph.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    This function uses a predefined numpy matrix to simulate the quantity
    distribution of fruits.

    Parameters:
        None

    Returns:
        None

    Example:
        >>> bars()
        # This will display the plot window with the stacked bar graph.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Define colors and labels corresponding to fruit types
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

    # Initialize the bottom position for the stacked bars
    bar_bottom = np.zeros(3)  # Starts with zero for the first fruit type

    # Loop through each fruit type and plot it as a part of the stacked bar
    for i in range(4):
        plt.bar(['Farrah', 'Fred', 'Felicia'], fruit[i], bottom=bar_bottom,
                color=colors[i], label=labels[i], width=0.5)
        bar_bottom += fruit[i]  # Update the start position for the next fruit

    # Configure plot settings
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.yticks(np.arange(0, 81, 10))

    plt.show()
