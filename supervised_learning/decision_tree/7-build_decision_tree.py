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

    def update_indicator(self):
        """
        Updates the indicator function for the node based on the bounds.
        This function defines whether an individual's features meet the
        node's criteria.
        """
        def is_large_enough(x):
            return np.array([np.greater_equal(x[:, key], self.lower[key])
                            for key in self.lower.keys()]).all(axis=0)

        def is_small_enough(x):
            return np.array([np.less_equal(x[:, key], self.upper[key])
                            for key in self.upper.keys()]).all(axis=0)

        self.indicator = lambda x: np.logical_and(is_large_enough(x),
                                                  is_small_enough(x))

    def update_predict(self):
        """
        Updates the prediction function of the decision tree.
        This function prepares the tree to make predictions by updating
        bounds, retrieving all leaves, and setting their indicators.
        It defines a lambda function as the predict method, which uses
        these indicators to determine which leaf's value to return for
        each input sample.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.array([leaf.value
                                           for x in A for leaf in leaves
                                           if leaf.indicator(x)])

    def pred(self, x):
        """
        Recursively predicts the value by navigating down the tree based
        on the input features.

        Args:
            x (array): The input features for a single sample.

        Returns:
            any: The predicted value from the child nodes.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def pred(self, x):
        """
        Predicts the value based on this leaf's stored value.

        Args:
            x (array): The input features for a single sample.

        Returns:
            any: The predicted value.
        """
        return self.value


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

    def update_bounds(self):
        """
        Updates bounds for all nodes in the tree starting from the root.
        """
        self.root.update_bounds_below()

    def update_indicator(self):
        """
        Updates the indicator functions for all nodes in the tree starting
        from the root.
        """
        self.root.update_indicator()

    def pred(self, x):
        """
        Predicts the value for a sample by delegating to the root node of
        the tree.

        Args:
            x (array): The input features for a single sample.

        Returns:
            any: The predicted value from the tree.
        """
        return self.root.pred(x)

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

    def fit(self, explanatory, target, verbose=0):
        """
        Fits the decision tree to the provided training data.

        Args:
            explanatory (array): Input features for the training data.
            target (array): Target values for the training data.
            verbose (int, optional): Verbosity level of the output.
            Default is 0.

        This method sets up the tree based on the chosen split criterion
        and recursively fits nodes, updating the tree's prediction function
        upon completion.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            # print("----------------------------------------------------")
            print(f"""  Training finished.
    - Depth                     : { self.depth()       }
    - Number of nodes           : { self.count_nodes() }
    - Number of leaves          : { self.count_nodes(only_leaves=True) }
    - Accuracy on training data : { self.accuracy(self.explanatory,
                                              self.target)}""")
            # print("----------------------------------------------------")

    def np_extrema(self, arr):
        """
        Computes the minimum and maximum values of an array.

        Args:
            arr (array): The input array.

        Returns:
            tuple: The minimum and maximum values in the array.
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

    def fit_node(self, node):
        """
        Recursively fits the tree starting from the given node.

        Args:
            node (Node): The node from which to start fitting the tree.

        This method splits the node if the conditions permit, or converts
        it into a leaf if the split conditions are not met (based on depth,
        population, or purity).
        """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = node.sub_population & (
                self.explanatory[:, node.feature] > node.threshold)
        right_population = node.sub_population & ~left_population

        # Is left node a leaf ?
        unique_classes, counts = np.unique(
            self.target[left_population], return_counts=True)
        is_left_leaf = (len(counts) == 1 or node.depth >= self.max_depth or
                        len(self.target[left_population]) < self.min_pop)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # Is right node a leaf ?
        unique_classes, counts = np.unique(
            self.target[right_population], return_counts=True)
        is_right_leaf = (len(counts) == 1 or node.depth >= self.max_depth or
                         len(self.target[right_population]) < self.min_pop)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Creates a leaf node from the given sub_population.

        Args:
            node (Node): The parent node from which the leaf is derived.
            sub_population (array): Subset of indices indicating the
            population for the leaf.

        Returns:
            Leaf: A new leaf node with a value determined by the most common
            class in sub_population.
        """
        target_values = self.target[sub_population]
        values, counts = np.unique(target_values, return_counts=True)
        value = values[np.argmax(counts)]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
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

    def accuracy(self, test_explanatory, test_target):
        """
        Computes the accuracy of the prediction model on test data.

        Args:
            test_explanatory (array): The explanatory variables of the test
            data.
            test_target (array): The target variables of the test data.

        Returns:
            float: The accuracy of the model on the test data, calculated as
            the ratio of correct predictions.
        """
        preds = self.predict(test_explanatory) == test_target
        return np.sum(preds) / len(test_target)
