from pymc3.bart.tree import Tree, SplitNode, LeafNode
from pymc3.model import Model
import numpy as np
from scipy import stats
from pymc3.bart.exceptions import (
    BARTParamsError,
)


class BART:
    def __init__(self, X, y, m=200, alpha=0.95, beta=2.0,
                 dist_splitting_variable='DiscreteUniform',
                 dist_splitting_rule_assignment='DiscreteUniform',
                 dist_leaf_given_structure='Normal',
                 sigma='InverseChiSquare'):

        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to "
                            "instantiate BART. Add variable "
                            "inside a 'with model:' block.")

        if not isinstance(X, np.ndarray) or X.dtype.type is not np.float64:
            raise BARTParamsError('The design matrix X type must be numpy.ndarray where every item'
                                  ' type is numpy.float64')
        if X.ndim != 2:
            raise BARTParamsError('The design matrix X must have two dimensions')
        if not isinstance(y, np.ndarray) or y.dtype.type is not np.float64:
            raise BARTParamsError('The response matrix y type must be numpy.ndarray where every item'
                                  ' type is numpy.float64')
        if y.ndim != 1:
            raise BARTParamsError('The response matrix y must have one dimension')
        if X.shape[0] != y.shape[0]:
            raise BARTParamsError('The design matrix X and the response matrix y must have the same number of elements')
        if not isinstance(m, int):
            raise BARTParamsError('The number of trees m type must be int')
        if m < 1:
            raise BARTParamsError('The number of trees m must be greater than zero')
        if not isinstance(alpha, float):
            raise BARTParamsError('The type for the alpha parameter for the tree structure must be float')
        if alpha <= 0 or 1 <= alpha:
            raise BARTParamsError('The value for the alpha parameter for the tree structure '
                                  'must be in the interval (0, 1)')
        if not isinstance(beta, float):
            raise BARTParamsError('The type for the beta parameter for the tree structure must be float')
        if beta < 0:
            raise BARTParamsError('The value for the beta parameter for the tree structure '
                                  'must be in the interval [0, float("inf"))')
        if dist_splitting_variable != 'DiscreteUniform':
            raise BARTParamsError('The distribution on the splitting variable assignments at each '
                                  'interior node must be "DiscreteUniform"')
        if dist_splitting_rule_assignment != 'DiscreteUniform':
            raise BARTParamsError('The distribution on the splitting rule assignment in each '
                                  'interior node must be "DiscreteUniform"')
        if dist_leaf_given_structure != 'Normal':
            raise BARTParamsError('The distribution on the leaves given the tree structure must '
                                  'be "Normal"')
        if sigma != 'InverseChiSquare':
            raise BARTParamsError('The distribution of the error variance must '
                                  'be "InverseChiSquare"')

        self.X = X
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.y = y

        self.type_variables = []
        for i in range(self.p):
            if self.is_variable_quantitative(i):
                self.type_variables.append('quantitative')
            else:
                self.type_variables.append('qualitative')

        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.dist_splitting_variable = dist_splitting_variable
        self.dist_splitting_rule_assignment = dist_splitting_rule_assignment
        self.dist_leaf_given_structure = dist_leaf_given_structure
        self.sigma = sigma

        self.y_min = self.y.min()
        self.y_max = self.y.max()
        self.y_min_y_max_half_diff = 0.5

        self.transformed_y = self.transform_y(self.y)

        initial_value_leaf_nodes = self.transformed_y.mean() / self.m
        initial_idx_data_points_leaf_nodes = np.array(range(self.n))
        self.trees = []
        for _ in range(self.m):
            new_tree = Tree.init_tree(leaf_node_value=initial_value_leaf_nodes,
                                      idx_data_points=initial_idx_data_points_leaf_nodes)
            self.trees.append(new_tree)

    def __repr__(self):

        representation = '''
BART(
     X = {},
     y = {},
     m = {},
     alpha = {},
     beta = {},
     dist_splitting_variable = {!r},
     dist_splitting_rule_assignment = {!r},
     dist_leaf_given_structure = {!r},
     sigma = {!r})'''
        return representation.format(type(self.X), type(self.y), self.m, self.alpha, self.beta,
                                     self.dist_splitting_variable, self.dist_splitting_rule_assignment,
                                     self.dist_leaf_given_structure, self.sigma)

    def transform_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min) - self.y_min_y_max_half_diff

    def un_transform_y(self, y):
        return (y + self.y_min_y_max_half_diff) * (self.y_max - self.y_min) + self.y_min

    def prediction_untransformed(self, x):
        sum_of_trees = 0.0
        for t in self.trees:
            sum_of_trees += t.out_of_sample_predict(x=x)
        return self.un_transform_y(sum_of_trees)

    def sample_dist_splitting_variable(self, value):
        if self.dist_splitting_variable == 'DiscreteUniform':
            return sample_from_discrete_uniform(0, value + 1)
        else:
            raise NotImplementedError

    def sample_dist_splitting_rule_assignment(self, value):
        if self.dist_splitting_rule_assignment == 'DiscreteUniform':
            return sample_from_discrete_uniform(0, value)
        else:
            raise NotImplementedError

    def get_available_predictors(self, idx_data_points_split_node):
        possible_splitting_variables = []
        for j in range(self.p):
            x_j = self.X[idx_data_points_split_node, j]
            for i in range(1, len(idx_data_points_split_node)):
                if x_j[i - 1] != x_j[i]:
                    possible_splitting_variables.append(j)
                    break
        return possible_splitting_variables

    def get_available_splitting_rules(self, idx_data_points_split_node, idx_split_variable):
        x_j = self.X[idx_data_points_split_node, idx_split_variable]
        values, indices = np.unique(x_j, return_index=True)
        return values, indices

    def grow_tree(self, tree, index_leaf_node):
        successful_grow_tree = False
        current_node = tree.get_node(index_leaf_node)

        available_predictors = self.get_available_predictors(current_node.idx_data_points)

        if not available_predictors:
            return successful_grow_tree

        index_selected_predictor = self.sample_dist_splitting_variable(len(available_predictors))
        selected_predictor = available_predictors[index_selected_predictor]

        available_splitting_rules, _ = self.get_available_splitting_rules(current_node.idx_data_points,
                                                                          selected_predictor)
        index_selected_splitting_rule = self.sample_dist_splitting_rule_assignment(len(available_splitting_rules))
        selected_splitting_rule = available_splitting_rules[index_selected_splitting_rule]

        new_split_node = SplitNode(index=index_leaf_node, idx_split_variable=selected_predictor,
                                   type_split_variable=self.type_variables[selected_predictor],
                                   split_value=selected_splitting_rule, idx_data_points=current_node.idx_data_points)

        # TODO: implement
        left_node_value = 0.0
        right_node_value = 0.0

        left_node_idx_data_points, right_node_idx_data_points = self.get_new_idx_data_points(new_split_node)

        new_left_node = LeafNode(index=current_node.get_idx_left_child(), value=left_node_value,
                                 idx_data_points=left_node_idx_data_points)
        new_right_node = LeafNode(index=current_node.get_idx_right_child(), value=right_node_value,
                                  idx_data_points=right_node_idx_data_points)
        tree.grow_tree(index_leaf_node, new_split_node, new_left_node, new_right_node)
        successful_grow_tree = True

        return successful_grow_tree

    def is_variable_quantitative(self, index_variable):
        # TODO: implement check to find out if a variable in self.X is quantitative or qualitative
        return True

    def get_new_idx_data_points_aux(self, current_split_node):
        # TODO: remove.
        # This function work for quantitative and qualitative data
        left_node_idx_data_points = []
        right_node_idx_data_points = []
        for i in current_split_node.idx_data_points:
            if current_split_node.evaluate_splitting_rule(self.X[i]):
                left_node_idx_data_points.append(i)
            else:
                right_node_idx_data_points.append(i)

        left_node_idx_data_points = np.array(left_node_idx_data_points)
        right_node_idx_data_points = np.array(right_node_idx_data_points)

        return left_node_idx_data_points, right_node_idx_data_points

    def get_new_idx_data_points(self, current_split_node):
        idx_data_points = current_split_node.idx_data_points
        idx_split_variable = current_split_node.idx_split_variable
        split_value = current_split_node.split_value

        left_idx = np.nonzero(self.X[idx_data_points, idx_split_variable] < split_value)
        left_node_idx_data_points = idx_data_points[left_idx]
        right_idx = np.nonzero(~(self.X[idx_data_points, idx_split_variable] < split_value))
        right_node_idx_data_points = idx_data_points[right_idx]

        return left_node_idx_data_points, right_node_idx_data_points


def sample_from_discrete_uniform(lower, upper, size=None):
        samples = stats.randint.rvs(lower, upper, size=size)
        return samples
