#!/usr/bin/env python3
"""Module for calculating the weighted moving average of a dataset."""


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a data set with bias correction.

    Parameters:
    data (list): List of data to calculate the moving average of.
    beta (float): The weight used for the moving average.

    Returns:
    list: List containing the moving averages of data.
    """
    v = 0
    moving_averages = []
    for i, value in enumerate(data):
        v = beta * v + (1 - beta) * value
        bias_correction = 1 - beta ** (i + 1)
        moving_avg = v / bias_correction
        moving_averages.append(moving_avg)
    return moving_averages
