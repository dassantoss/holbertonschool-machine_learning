#!/usr/bin/env python3
"""
This Module Plot a figure with multiple subplots showcasing different
types of plots.
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    This function generates a figure with 3x2 grid of subplots, each
    containing a different plot (line plot, scatter plot, etc.).
    The last subplot is removed to allow the histogram to take two
    columns widths.

    Parameters:
    None

    Returns:
    None. Displays a matplotlib figure.

    Example:
    >>> all_in_one()
    # This will display the figure with subplots in a matplotlib viewer.
    """
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Start plotting the subplots
    plt.figure(figsize=(9, 6))

    # Plot 1: Simple cubic line plot
    plt.subplot(321)
    plt.plot(y0, color="red")
    plt.xlim(0, 10)

    # Plot 2: Scatter plot for men's height vs weight
    plt.subplot(322)
    plt.scatter(x1, y1, color="magenta")
    plt.xlabel("Height (in)", fontsize="x-small")
    plt.ylabel("Weight (lbs)", fontsize="x-small")
    plt.title("Men's Height vs Weight", fontsize="x-small")

    # Plot 3: Exponential decay of C-14
    plt.subplot(323)
    plt.plot(x2, y2)
    plt.xlabel("Time (years)", fontsize="x-small")
    plt.ylabel("Fraction Remaining", fontsize="x-small")
    plt.title("Exponential Decay of C-14", fontsize="x-small")
    plt.yscale("log")
    plt.xlim(0, 28650)

    # Plot 4: Exponential decay of C-14 and Ra-226
    plt.subplot(324)
    plt.plot(x3, y31, 'r--', label='C-14')
    plt.plot(x3, y32, 'g-', label='Ra-226')
    plt.xlabel("Time (years)", fontsize="x-small")
    plt.ylabel("Fraction Remaining", fontsize="x-small")
    plt.title("Exponential Decay of Radioactive Elements", fontsize="x-small")
    plt.legend(fontsize='x-small')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Plot 5: Histogram of student grades
    plt.subplot(313)
    plt.hist(student_grades, bins=10, range=(0, 100), edgecolor="black")
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)
    plt.ylim(0, 30)
    plt.xlabel("Grades", fontsize="x-small")
    plt.ylabel("Number of Students", fontsize="x-small")
    plt.title("Project A", fontsize="x-small")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Set the overall title for the figure
    plt.suptitle("All in One", fontsize='x-small', verticalalignment='top')

    # Display the figure with all subplots
    plt.show()
