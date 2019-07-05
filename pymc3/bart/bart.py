from pymc3.bart.tree import Tree
from pymc3.model import Model
import numpy as np
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
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.dist_splitting_variable = dist_splitting_variable
        self.dist_splitting_rule_assignment = dist_splitting_rule_assignment
        self.dist_leaf_given_structure = dist_leaf_given_structure
        self.sigma = sigma

        self.y_min = self.y.min()
        self.y_max = self.y.max()
        self.initial_point_y_transformed = 0.5

        self.transformed_y = self.transform_y(self.y)

        initial_value_leaf_nodes = self.transformed_y.mean() / self.m
        self.trees = []
        for _ in range(self.m):
            new_tree = Tree.init_tree(initial_value_leaf_nodes)
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

    def __str__(self):
        return 'BART first tree:\n{}'.format(self.trees[0].__str__())

    def transform_y(self, y):
        return (y - self.y_min) / (self.y_max - self.y_min) - self.initial_point_y_transformed

    def un_transform_y(self, y):
        return (y + self.initial_point_y_transformed) * (self.y_max - self.y_min) + self.y_min

    def prediction_untransformed(self, x):
        sum_of_trees = 0.0
        for t in self.trees:
            sum_of_trees += t.out_of_sample_predict(x=x)
        return self.un_transform_y(sum_of_trees)
