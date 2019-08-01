from pymc3.bart.tree import Tree, SplitNode, LeafNode
from pymc3.model import Model
import numpy as np
from scipy import stats
from pymc3.bart.exceptions import (
    BARTParamsError,
)


class BART:
    def __init__(self, X, Y, m=200, alpha=0.95, beta=2.0,
                 nu=3.0,
                 q=0.9,
                 k=2.0,
                 tree_sampler='GrowPrune'):

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
        if not isinstance(Y, np.ndarray) or Y.dtype.type is not np.float64:
            raise BARTParamsError('The response matrix y type must be numpy.ndarray where every item'
                                  ' type is numpy.float64')
        if Y.ndim != 1:
            raise BARTParamsError('The response matrix Y must have one dimension')
        if X.shape[0] != Y.shape[0]:
            raise BARTParamsError('The design matrix X and the response matrix Y must have the same number of elements')
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
        # TODO: in the future these checks will change
        if not isinstance(nu, float):
            raise BARTParamsError('The type for the nu parameter related to the sigma prior must be float')
        if nu < 3.0:
            raise BARTParamsError('Chipman et al. discourage the use of nu less than 3.0')
        if not isinstance(q, float):
            raise BARTParamsError('The type for the q parameter related to the sigma prior must be float')
        if q <= 0 or 1 <= q:
            raise BARTParamsError('The value for the q parameter related to the sigma prior '
                                  'must be in the interval (0, 1)')
        if not isinstance(k, float):
            raise BARTParamsError('The type for the k parameter related to the mu_ij given T_j prior must be float')
        if k <= 0:
            raise BARTParamsError('The value for the k parameter k parameter related to the mu_ij given T_j prior '
                                  'must be in the interval (0, float("inf"))')
        if tree_sampler not in ['GrowPrune', 'ParticleGibbs']:
            raise BARTParamsError('{} is not a valid tree sampler'.format(tree_sampler))

        self.X = X
        self.num_observations = X.shape[0]
        self.number_variates = X.shape[1]
        self.Y = Y

        self.prior_k = k

        self.Y_min = self.Y.min()
        self.Y_max = self.Y.max()
        self.Y_transf_max_Y_transf_min_half_diff = 3.0 if self.check_if_is_binary_classification() else 0.5

        self.Y_transformed = self.transform_Y(self.Y)

        self.m = m
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.prior_nu = nu
        self.prior_q = q
        self.overestimated_sigma = self.Y_transformed.std()
        self.prior_lambda = compute_lambda_value_scaled_inverse_chi_square(self.overestimated_sigma,
                                                                           self.prior_q, self.prior_nu)
        self.current_sigma = 1.0

        self.prior_mu_mu = 0.0
        self.prior_sigma_mu = self.Y_transf_max_Y_transf_min_half_diff / (self.prior_k * np.sqrt(self.m))

        self.tree_sampler = tree_sampler

        self.are_priors_conjugate = self.check_if_priors_are_conjugate()
        self.debug_flexible_implementation = False

        # Diff trick to speed computation of residuals.
        # Taken from Section 3.1 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
        # The sum_trees_output will contain the sum of the predicted output for all trees.
        # When R_j is needed we subtract the current predicted output for tree T_j.
        self.sum_trees_output = np.zeros_like(self.Y)

        initial_value_leaf_nodes = self.Y_transformed.mean() / self.m
        initial_idx_data_points_leaf_nodes = np.array(range(self.num_observations), dtype='int64')
        self.trees = []
        for _ in range(self.m):
            new_tree = Tree.init_tree(leaf_node_value=initial_value_leaf_nodes,
                                      idx_data_points=initial_idx_data_points_leaf_nodes)
            self.trees.append(new_tree)

    def __repr__(self):

        representation = '''
BART(
     X = {},
     Y = {},
     m = {},
     alpha = {},
     beta = {},
     nu = {},
     q = {},
     k = {})'''
        return representation.format(type(self.X), type(self.Y), self.m, self.prior_alpha, self.prior_beta,
                                     self.prior_nu, self.prior_q, self.prior_k)

    def check_if_priors_are_conjugate(self):
        # TODO: check if the likelihood is normal and the sigma prior is scaled inverse chi-square
        return True

    def check_if_is_binary_classification(self):
        # TODO: check if the user is doing binary classification or regression
        return False

    def transform_Y(self, Y):
        return (Y - self.Y_min) / (self.Y_max - self.Y_min) - self.Y_transf_max_Y_transf_min_half_diff

    def un_transform_Y(self, Y):
        return (Y + self.Y_transf_max_Y_transf_min_half_diff) * (self.Y_max - self.Y_min)\
               / (self.Y_transf_max_Y_transf_min_half_diff * 2) + self.Y_min

    def prediction_untransformed(self, x):
        sum_of_trees = 0.0
        for t in self.trees:
            sum_of_trees += t.out_of_sample_predict(x=x)
        return self.un_transform_Y(sum_of_trees)

    def sample_dist_splitting_variable(self, value):
        return sample_from_discrete_uniform(0, value + 1)

    def sample_dist_splitting_rule_assignment(self, value):
        # The last value is not consider since if we choose it as the value of
        # the splitting rule assignment, it would leave the right subtree empty.
        return sample_from_discrete_uniform(0, value)

    def get_available_predictors(self, idx_data_points_split_node):
        possible_splitting_variables = []
        for j in range(self.number_variates):
            x_j = self.X[idx_data_points_split_node, j]
            x_j = x_j[~np.isnan(x_j)]
            for i in range(1, len(x_j)):
                if x_j[i - 1] != x_j[i]:
                    possible_splitting_variables.append(j)
                    break
        return possible_splitting_variables

    def get_available_splitting_rules(self, idx_data_points_split_node, idx_split_variable):
        x_j = self.X[idx_data_points_split_node, idx_split_variable]
        x_j = x_j[~np.isnan(x_j)]
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
                                   type_split_variable='quantitative',
                                   split_value=selected_splitting_rule, idx_data_points=current_node.idx_data_points)

        left_node_idx_data_points, right_node_idx_data_points = self.get_new_idx_data_points(new_split_node)

        left_node_value = self.draw_leaf_value(tree, left_node_idx_data_points)
        right_node_value = self.draw_leaf_value(tree, right_node_idx_data_points)

        new_left_node = LeafNode(index=current_node.get_idx_left_child(), value=left_node_value,
                                 idx_data_points=left_node_idx_data_points)
        new_right_node = LeafNode(index=current_node.get_idx_right_child(), value=right_node_value,
                                  idx_data_points=right_node_idx_data_points)
        tree.grow_tree(index_leaf_node, new_split_node, new_left_node, new_right_node)
        successful_grow_tree = True

        return successful_grow_tree

    def prune_tree(self, tree, index_split_node):
        current_node = tree.get_node(index_split_node)

        leaf_node_value = self.draw_leaf_value(tree, current_node.idx_data_points)

        new_leaf_node = LeafNode(index=index_split_node, value=leaf_node_value,
                                 idx_data_points=current_node.idx_data_points)
        tree.prune_tree(index_split_node, new_leaf_node)

    def get_new_idx_data_points(self, current_split_node):
        idx_data_points = current_split_node.idx_data_points
        idx_split_variable = current_split_node.idx_split_variable
        split_value = current_split_node.split_value

        left_idx = np.nonzero(self.X[idx_data_points, idx_split_variable] < split_value)
        left_node_idx_data_points = idx_data_points[left_idx]
        right_idx = np.nonzero(~(self.X[idx_data_points, idx_split_variable] < split_value))
        right_node_idx_data_points = idx_data_points[right_idx]

        return left_node_idx_data_points, right_node_idx_data_points

    def draw_leaf_value(self, tree, idx_data_points):
        # Method extracted from the function LeafNodeSampler.sample() of bartpy
        current_num_observations = len(idx_data_points)
        R_j = self.get_residuals(tree)
        node_responses = R_j[idx_data_points]
        node_average_responses = np.sum(node_responses) / current_num_observations

        if self.are_priors_conjugate and not self.debug_flexible_implementation:
            prior_var = self.prior_sigma_mu ** 2
            likelihood_var = (self.current_sigma ** 2) / current_num_observations
            likelihood_mean = node_average_responses
            posterior_variance = 1. / (1. / prior_var + 1. / likelihood_var)
            posterior_mean = likelihood_mean * (prior_var / (likelihood_var + prior_var))
            # TODO: the samples of the normal can be cached for improved performance like
            #  bartpy does in the class NormalScalarSampler
            draw = posterior_mean + (np.random.normal() * np.power(posterior_variance / self.m, 0.5))
            return draw
        else:
            raise NotImplementedError()

    def get_residuals(self, tree):
        R_j = self.sum_trees_output - tree.predict_output(self.num_observations)
        return R_j

    def draw_sigma_from_posterior(self):
        # Method extracted from the function SigmaSampler.sample() of bartpy
        if self.are_priors_conjugate and not self.debug_flexible_implementation:
            posterior_alpha = self.prior_nu + (self.num_observations / 2.)
            quadratic_error = np.sum(np.square(self.Y_transformed - self.sum_trees_output))
            posterior_beta = self.prior_lambda + (0.5 * quadratic_error)
            draw = np.power(np.random.gamma(posterior_alpha, 1. / posterior_beta), -0.5)
            return draw
        else:
            raise NotImplementedError()

    def sample_tree_structure(self):
        if self.tree_sampler == 'GrowPrune':
            print()
        elif self.tree_sampler == 'PG':
            print()
        else:
            print('error')


def sample_from_discrete_uniform(lower, upper, size=None):
        samples = stats.randint.rvs(lower, upper, size=size)
        return samples


def compute_lambda_value_scaled_inverse_chi_square(overestimated_sigma, q, nu):
    # Method extracted from the function calculateHyperparameters() of bartMachine
    return stats.distributions.chi2.ppf(1 - q, nu) / nu * overestimated_sigma
