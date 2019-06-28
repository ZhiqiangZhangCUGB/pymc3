from pymc3.bart.tree import Tree, SplitNode, LeafNode
from copy import copy


class BART:
    def __init__(self, X, Y, m=200, alpha=0.95, beta=2,
                 dist_splitting_variable=None,
                 dist_splitting_rule_assignment=None,
                 dist_leaf_given_structure=None,
                 sigma=None):
        initial_value_leaf_nodes = Y.mean() / m
        initial_tree = Tree.init_tree(initial_value_leaf_nodes)
        self.trees = [initial_tree]
        for _ in range(m - 1):
            new_tree = copy(initial_tree)
            self.trees.append(new_tree)
