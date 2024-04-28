#!/usr/bin/env python3
"""
Module implementing Isolation_Random_Forest for outlier detection using
Isolation Trees.
Designed for high-dimensional datasets, it identifies anomalies based on
data splits by feature selection.
"""
import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    A class that implements an Isolation Forest for anomaly detection.
    An Isolation Forest consists of multiple Isolation Random Trees which
    isolate observations by randomly selecting a feature and then randomly
    selecting a split value between the maximum and minimum values of the
    selected feature.

    Attributes:
        n_trees (int): Number of trees in the forest.
        max_depth (int): The maximum depth of each tree in the forest.
        min_pop (int): Minimum sample size in nodes below which a tree
        will not attempt to split.
        seed (int): Random seed used for reproducible results.
        numpy_preds (list): List of prediction functions from each tree in
        the forest.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the Isolation Random Forest with specified parameters.

        Args:
            n_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth of each tree.
            min_pop (int): Minimum population size in a node to consider
            for further splitting.
            seed (int): Seed for the random number generator to ensure
            reproducibility.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts the anomaly scores for each sample in the dataset.

        Args:
            explanatory (numpy.ndarray): Data to predict the anomaly
            scores for.

        Returns:
            numpy.ndarray: Averaged depth of each sample across all trees
            in the forest.
        """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        mean_predictions = predictions.mean(axis=0)
        if self.verbose:
            print("Predictions Mean per Sample:", mean_predictions)
        return mean_predictions

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fits the Isolation Forest model to the provided data.

        Args:
            explanatory (numpy.ndarray): The dataset to fit the model.
            n_trees (int, optional): Number of trees to generate in the
            forest.
            verbose (int, optional): Verbosity level; 0 is silent,
            1 prints the tree statistics after training.

        Returns:
            None
        """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        for i in range(n_trees):
            T = Isolation_Random_Tree(max_depth=self.max_depth,
                                      seed=self.seed + i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : { np.array(depths).mean()      }
    - Mean number of nodes           : { np.array(nodes).mean()       }
    - Mean number of leaves          : { np.array(leaves).mean()      }""")

    def suspects(self, explanatory, n_suspects):
        """
        Identifies the top n suspects with the lowest mean depth in the
        Isolation Forest, suggesting they are potential outliers.

        Args:
            explanatory (numpy.ndarray): The explanatory variables of the
            dataset.
            n_suspects (int): The number of suspect data points to return.

        Returns:
            tuple: Two numpy arrays; the first contains the suspect data
            points,
                the second contains the corresponding depths indicating
                their isolation levels.
        """
        depths = self.predict(explanatory)
        sorted_indices = np.argsort(depths)
        return explanatory[sorted_indices[:n_suspects]], \
            depths[sorted_indices[:n_suspects]]
