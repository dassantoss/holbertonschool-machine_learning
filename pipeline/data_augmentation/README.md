# Data Augmentation

## Overview

This project is part of the Machine Learning curriculum at Holberton School, focusing on data augmentation techniques for image processing using TensorFlow. The project covers various image transformation methods essential for improving machine learning model performance and robustness.

## Learning Objectives

By the end of this project, you should be able to explain:

- What data augmentation is and its importance
- When to perform data augmentation
- Benefits of using data augmentation
- Various ways to perform data augmentation
- How to use ML to automate data augmentation

## Requirements

### General
- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- TensorFlow 2.15
- TensorFlow Datasets 4.9.2
- Pycodestyle 2.11.1

### Installation
```bash
pip install --user tensorflow-datasets==4.9.2
```

## Project Tasks

1. **Flip Image**: Implement horizontal image flipping
2. **Crop Image**: Perform random image cropping
3. **Rotate Image**: Rotate images 90 degrees counter-clockwise
4. **Contrast Adjustment**: Implement random contrast adjustment
5. **Brightness Adjustment**: Apply random brightness changes
6. **Hue Modification**: Implement hue adjustments
7. **PCA Color Augmentation**: Advanced color augmentation using PCA

## Resources

- [Data Augmentation: train deep learning models with less data](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/)
- [A Complete Guide to Data Augmentation](https://www.v7labs.com/blog/data-augmentation-guide)
- [TensorFlow Image Documentation](https://www.tensorflow.org/api_docs/python/tf/image)
- [Image Data Augmentation using TensorFlow](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [Automating Data Augmentation](https://arxiv.org/abs/1909.13719)

## Repository Structure

```
holbertonschool-machine_learning/
└── pipeline/
    └── data_augmentation/
        ├── 0-flip.py
        ├── 1-crop.py
        ├── 2-rotate.py
        ├── 3-contrast.py
        ├── 4-brightness.py
        ├── 5-hue.py
        ├── 100-pca.py
        └── README.md
```

## Author
Project developed as part of the Machine Learning curriculum at Holberton School.

## License
This project is licensed under the terms of the MIT license.