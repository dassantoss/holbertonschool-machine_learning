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

    def __str__(self):
        """
        Provides a string representation of the node and its children.

        Returns:
            str: A string representation of the subtree rooted at this node.
        """
        p = "root" if self.is_root else "-> node"
        result = f"{p} [feature={self.feature},\
 threshold={self.threshold}]\n"
        if self.left_child:
            result +=\
                self.left_child_add_prefix(self.left_child.__str__().strip())
        if self.right_child:
            result +=\
                self.right_child_add_prefix(self.right_child.__str__().strip())
        return result

    def left_child_add_prefix(self, text):
        """
        Adds a prefix for the left child's subtree representation.

        Args:
            text (str): The subtree string.

        Returns:
            str: The modified subtree string with added prefixes.
        """
        lines = text.split("\n")
        new_text = "    +--"+lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("    |  "+x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        Adds a prefix for the right child's subtree representation.

        Args:
            text (str): The subtree string.

        Returns:
            str: The modified subtree string with added prefixes.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

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

    def count_nodes_below(self, only_leaves=False):
        """
        Counts the nodes below this node, optionally counting only the leaves.

        Args:
            only_leaves (bool): Whether to count only leaves.

        Returns:
            int: The count of nodes or leaves below this node.
        """
        if only_leaves:
            if self.is_leaf:
                return 1
            count = 0
        else:
            count = 1  # Count this node

        if self.left_child:
            count += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            count += self.right_child.count_nodes_below(only_leaves)
        return count

    def get_leaves_below(self):
        """
        Retrieves all leaf nodes below this node.

        Returns:
            list: A list of all leaf nodes below this node.
        """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            if self.left_child:
                leaves.extend(self.left_child.get_leaves_below())
            if self.right_child:
                leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively updates bounds for this node and its children.
        Initializes at root with infinite bounds and adjusts for children
        based on data.
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                # Make a copy of the current node's bounds to each child.
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                elif child == self.right_child:
                    child.upper[self.feature] = self.threshold
        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


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

    def __str__(self):
        """
        Provides a string representation of the leaf.

        Returns:
            str: A string representation of this leaf.
        """
        return (f"-> leaf [value={self.value}] ")

    def max_depth_below(self):
        """
        Returns the depth of the leaf, as leaves are the end of a branch.

        Returns:
            int: The depth of this leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Returns the count of this node as a leaf, regardless of
        the only_leaves flag.

        Args:
            only_leaves (bool): Ignored in leaf context as a leaf
            is always counted.

        Returns:
            int: Always 1, since a leaf counts as one node.
        """
        return 1

    def get_leaves_below(self):
        """
        Since this node is a leaf, it returns itself in a list.

        Returns:
            list: A list containing only this leaf.
        """
        return [self]

    def update_bounds_below(self):
        """
        Leaves do not update bounds, so this is a placeholder.
        """
        pass


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
