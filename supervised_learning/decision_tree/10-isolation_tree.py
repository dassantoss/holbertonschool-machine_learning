#!/usr/bin/env python3
"""
Module for the Isolation_Random_Tree class. It's designed for detecting
anomalies by isolating data points in a tree structure, where anomalies
are isolated closer to the root of the tree. It works without prior
knowledge of the distribution of normal data points.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """
    Represents an isolation tree used for anomaly detection. It is similar
    to a decision tree but is used for identifying outliers by isolating
    samples.

    Attributes:
        rng (np.random.Generator): Random number generator.
        root (Node): The root node of the isolation tree.
        explanatory (np.ndarray): The explanatory feature data for fitting.
        max_depth (int): The maximum depth of the tree.
        predict (callable): The prediction function of the isolation tree.
        min_pop (int): The minimum population for splitting during training.
    """
    def __init__(self, max_depth=10, seed=0, root=None):
        """
        Initializes the Isolation_Random_Tree with the given max_depth and
        seed.

        Args:
            max_depth (int): The maximum depth that the tree can grow to.
            seed (int): Seed for the random number generator to ensure
            reproducibility.
            root (Node, optional): The root node of the tree if starting
            from an existing node.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """
        Provides a string representation of the entire decision tree.

        Returns:
            str: A string representation of the decision tree.
        """
        return self.root.__str__()

    def depth(self):
        """
        Computes the maximum depth of the tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Counts the nodes in the entire tree, with an option
        to count only leaves.

        Args:
            only_leaves (bool): Whether to count only leaves.

        Returns:
            int: Total number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Retrieves all leaves in the decision tree.

        Returns:
            list: A list of all leaf nodes in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Starts the recursive update of bounds from the root.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Updates the prediction function for the decision tree.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([
            next(leaf.value for leaf in leaves
                 if leaf.indicator(x.reshape(1, -1)))
            for x in np.atleast_2d(A)
        ])

    def np_extrema(self, arr):
        """
        Returns the minimum and maximum values from the array.

        Args:
            arr (np.ndarray): The input array.

        Returns:
            tuple: A tuple containing the minimum and maximum values.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Determines a random splitting criterion for a node based
        on feature values.

        Args:
            node (Node): The node for which to determine the split.

        Returns:
            tuple: The chosen feature index and threshold value for
            the split.
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """
        Creates and returns a leaf child node.

        Args:
            node (Node): The parent node from which the leaf is derived.
            sub_population (np.ndarray): The subset of indices that belong
            to this leaf.

        Returns:
            Leaf: The created leaf child node.
        """
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Creates a new child node for further splits.

        Args:
            node (Node): The parent node.
            sub_population (array): Subset of indices for the new node's
            population.

        Returns:
            Node: A new child node initialized for further splitting.
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """
        Recursively fits the isolation tree starting from a given node by
        isolating samples at each split until the maximum depth is reached
        or there are no more samples to isolate.

        Args:
            node (Node): The node to fit.
        """
        node.feature, node.threshold = self.random_split_criterion(node)

        above_threshold = self.explanatory[:, node.feature] > node.threshold
        left_population = node.sub_population & above_threshold
        right_population = node.sub_population & ~above_threshold

        # Check if the left or right node should be a leaf
        is_left_leaf = np.any([
            node.depth >= self.max_depth - 1,
            np.sum(left_population) <= self.min_pop
        ])

        # Create left child
        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Check if the right node should be a leaf
        is_right_leaf = np.any([
            node.depth >= self.max_depth - 1,
            np.sum(right_population) <= self.min_pop
        ])

        # Create right child
        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """
        Fits the isolation tree to the given data.

        Args:
            explanatory (np.ndarray): The data to fit the tree to.
            verbose (int): Indicates the verbosity level for logging
            the training process.
        """
        self.split_criterion = self.random_split_criterion
        self.explanatory = explanatory
        self.root.sub_population = np.ones_like(explanatory.shape[0],
                                                dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }""")
