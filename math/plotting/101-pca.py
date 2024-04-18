#!/usr/bin/env python3
"""
This module performs Principal Component Analysis (PCA) on the Iris dataset
and visualizes the results in a 3D scatter plot, color-coded by species.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_pca_iris():
    """
    Function that Visualizes the Iris dataset using PCA in a 3D scatter plot.

    Parameters:
    None

    Returns:
    None
    """
    # Load the data and labels from the .npy files
    data = np.load('data.npy')
    labels = np.load('labels.npy')

    # Subtract the mean from the data
    data_means = np.mean(data, axis=0)
    norm_data = data - data_means

    # Perform SVD
    _, _, Vh = np.linalg.svd(norm_data)

    # Project the data onto the first three principal components
    pca_data = np.matmul(norm_data, Vh[:3].T)

    # Initialize 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points in the PCA space
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
                         c=labels, cmap='plasma')

    # Label the axes
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    ax.set_zlabel('U3')

    # Set the title
    plt.title('PCA of Iris Dataset')

    # Create color bar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Species')

    # Display the plot
    plt.show()


if __name__ == "__main__":
    plot_pca_iris()
