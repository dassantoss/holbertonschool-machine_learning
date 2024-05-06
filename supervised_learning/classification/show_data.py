#!/usr/bin/env python3
"""
This script loads a dataset of labeled binary images and displays the first 100 images in a 10x10 grid.
Each subplot shows an image with its respective label as a title, and all axes are turned off for clarity.

Dependencies:
    matplotlib.pyplot: For creating visualizations.
    numpy: For handling arrays.

Data files:
    ../data/Binary_Train.npz: A compressed NumPy file containing 'X' and 'Y', 
                              where 'X' is the array of images and 'Y' is the corresponding labels.

"""
# Load data from a .npz file
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with specific size
lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
