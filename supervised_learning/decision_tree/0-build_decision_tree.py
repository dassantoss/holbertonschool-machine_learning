#!/usr/bin/env python3
"""
This module defines the classes for building a basic decision tree,
including Node, Leaf, and Decision_Tree.
"""
import numpy as np


class Node:
    """
    Represents a node in a decision tree.

    Attributes:
        feature (int): Index of the feature used for splitting.
        threshold (float): Threshold value for the split.
        left_child (Node): Left child node.
        right_child (Node): Right child node.
        is_root (bool): Indicates if the node is the root.
        depth (int): Depth of the node in the tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Calculates the maximum depth below this node.

        Returns:
            int: Maximum depth below this node.
        """
        max_depth = self.depth
        if self.left_child:
            max_depth = max(max_depth, self.left_child.max_depth_below())
        if self.right_child:
            max_depth = max(max_depth, self.right_child.max_depth_below())
        return max_depth


class Leaf(Node):
    """
    Represents a leaf in a decision tree.

    Attributes:
        value (any): The value predicted by this leaf.
        depth (int): Depth of the leaf in the tree.
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf, as leaves are the end of a branch.

        Returns:
            int: The depth of this leaf.
        """
        return self.depth


class Decision_Tree():
    """
    Represents a decision tree.

    Attributes:
        max_depth (int): Maximum depth of the tree.
        min_pop (int): Minimum population required to split a node.
        seed (int): Seed for the random number generator.
        split_criterion (str): Criterion used for splitting nodes.
        root (Node): The root node of the tree.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Computes the maximum depth of the tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()
