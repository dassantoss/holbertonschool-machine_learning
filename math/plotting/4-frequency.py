#!/usr/bin/env python3
"""
This Module Plot a histogram of student grades for a project.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    This function creates a histogram representing the distribution of student
    grades.  The histogram's bars are outlined in black, with bins every 10
    units to categorize the grades.  The plot is tittled 'Project A', with
    de x-axis labeled 'Grades' and the y-axis labeled 'number of Students'.

    Parameters:
    None

    Returns:
    None. Displays a matplotlib figure.

    Example:
    >>> frequency()
    # This will display the histogram in a matplotlib viewer.
    """
    # Seed the random number generator for reproducibility
    np.random.seed(5)
    # Generate student grades data
    student_grades = np.random.normal(68, 15, 50)
    # Set the figure size
    plt.figure(figsize=(6.4, 4.8))

    # Plot the histogram
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xticks(range(0, 101, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 30)

    # Set the title and labels
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')

    # Show the plot
    plt.show()
