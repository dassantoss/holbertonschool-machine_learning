#!/usr/bin/env python3
"""
This script loads the MNIST dataset from a .npz file and displays
the first 100 training images in a 10x10 grid.
Each image is displayed with its corresponding label as the title.
All axes are hidden to emphasize the images.

Dependencies:
    matplotlib.pyplot: For creating visualizations.
    numpy: For handling arrays and data manipulation.

Data files:
    ../data/MNIST.npz: A compressed NumPy file containing 'X_train'
    and 'Y_train',
                       where 'X_train' is the array of training images
                       and 'Y_train' is the corresponding labels.
"""

import matplotlib.pyplot as plt
import numpy as np

# Load data from a .npz file
lib = np.load('../data/MNIST.npz')
print(lib.files)  # Output the available arrays in the dataset
X_train_3D = lib['X_train']  # Load training images
Y_train = lib['Y_train']  # Load training labels

# Create a figure with a specified size
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_train_3D[i], cmap='gray')  # Display image in grayscale
    plt.title(str(Y_train[i]))  # Set title as the label
    plt.axis('off')  # Hide axes
plt.tight_layout()  # Adjust subplots to fit into the figure area
plt.show()  # Display the figure
