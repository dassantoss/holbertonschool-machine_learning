#!/usr/bin/env python3
"""
This module provides a Random Forest classifier that uses Decision Trees
as its base learners. It includes methods for fitting the model to training
data, making predictions, and calculating accuracy.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
    Random Forest classifier using Decision Trees as base learners.

    This class implements a Random Forest classifier using Decision Trees
    as its base learners. It includes methods for fitting the model to training
    data, making predictions, and calculating accuracy.

    Attributes:
        numpy (module): Numerical Python library for array operations.
        Decision_Tree (module): Decision Tree class from the
        "8-build_decision_tree" module.

    Example Usage:
        rf = Random_Forest()
        rf.fit(train_explanatory, train_target)
        predictions = rf.predict(test_explanatory)
        acc = rf.accuracy(test_explanatory, test_target)

    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes a Random_Forest classifier.

        Args:
            n_trees (int): Number of trees in the forest (default is 100).
            max_depth (int): Maximum depth of each decision tree
            (default is 10).
            min_pop (int): Minimum samples required to split a node
            default is 1).
            seed (int): Seed for random number generation (default is 0).
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the target variable for input data.

        Args:
            explanatory (numpy.ndarray): Input data for prediction.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        predictions = []

        # Generate predictions for each tree in the forest
        for predict_function in self.numpy_preds:
            predictions.append(predict_function(explanatory))

        predictions = np.array(predictions)

        # Calculate the mode (most frequent) prediction for each example
        mode_predictions = []
        for example_predictions in predictions.T:
            unique_values, counts = np.unique(example_predictions,
                                              return_counts=True)
            mode_index = np.argmax(counts)
            mode_predictions.append(unique_values[mode_index])

        return np.array(mode_predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Fits the Random Forest classifier to training data.

        Args:
            explanatory (numpy.ndarray): Input features for training.
            target (numpy.ndarray): Target variable for training.
            n_trees (int): Number of trees in the forest (default is 100).
            verbose (int): Verbosity level (0 for silent, 1 for detailed
            output).

        Returns:
            None
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth, min_pop=self.min_pop,
                              seed=self.seed+i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")
            print(f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory,self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """
        Calculates the accuracy of the model on test data.

        Args:
            test_explanatory (numpy.ndarray): Input features for testing.
            test_target (numpy.ndarray): Target variable for testing.

        Returns:
            float: Accuracy score.
        """
        return np.sum(np.equal(self.predict(test_explanatory),
                      test_target))/test_target.size
