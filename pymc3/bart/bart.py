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

        self.type_variables = []
        for i in range(self.number_variates):
            if self.is_variable_quantitative(i):
                self.type_variables.append('quantitative')
            else:
                self.type_variables.append('qualitative')

        self.prior_k = k

        self.Y_min = self.Y.min()
        self.Y_max = self.Y.max()
        self.Y_min_Y_max_half_diff = 0.5

        self.transformed_Y = self.transform_Y(self.Y)

        self.m = m
        self.prior_alpha = alpha
        self.prior_beta = beta
        self.prior_nu = nu
        self.prior_q = q
        self.overestimated_sigma = self.transformed_Y.var()
        self.prior_lambda = compute_lambda_value_scaled_inverse_chi_square(self.overestimated_sigma,
                                                                           self.prior_q, self.prior_nu)
        self.current_sigma_square = 1.0

        # TODO: Assign correct values
        self.prior_mu_mu = 0.0
        self.prior_sigma_square_mu = np.power(self.Y_min_Y_max_half_diff / (self.prior_k * np.sqrt(self.m)), 2)

        self.tree_sampler = tree_sampler

        # Diff trick to speed computation of residuals.
        # Taken from Section 3.1 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
        # The residuals_vector will contain the predicted output for all trees.
        # When R_j is needed we subtract the current predicted output for tree T_j.
        self.residuals_vector = np.zeros_like(self.Y)

        initial_value_leaf_nodes = self.transformed_Y.mean() / self.m
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

    def transform_Y(self, Y):
        return (Y - self.Y_min) / (self.Y_max - self.Y_min) - self.Y_min_Y_max_half_diff

    def un_transform_Y(self, Y):
        return (Y + self.Y_min_Y_max_half_diff) * (self.Y_max - self.Y_min) + self.Y_min

    def prediction_untransformed(self, x):
        sum_of_trees = 0.0
        for t in self.trees:
            sum_of_trees += t.out_of_sample_predict(x=x)
        return self.un_transform_Y(sum_of_trees)

    def sample_dist_splitting_variable(self, value):
        return sample_from_discrete_uniform(0, value + 1)

    def sample_dist_splitting_rule_assignment(self, value):
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
                                   type_split_variable=self.type_variables[selected_predictor],
                                   split_value=selected_splitting_rule, idx_data_points=current_node.idx_data_points)

        left_node_idx_data_points, right_node_idx_data_points = self.get_new_idx_data_points(new_split_node)

        left_node_value = self.sample_leaf_value(left_node_idx_data_points)
        right_node_value = self.sample_leaf_value(right_node_idx_data_points)

        new_left_node = LeafNode(index=current_node.get_idx_left_child(), value=left_node_value,
                                 idx_data_points=left_node_idx_data_points)
        new_right_node = LeafNode(index=current_node.get_idx_right_child(), value=right_node_value,
                                  idx_data_points=right_node_idx_data_points)
        tree.grow_tree(index_leaf_node, new_split_node, new_left_node, new_right_node)
        successful_grow_tree = True

        return successful_grow_tree

    def prune_tree(self, tree, index_split_node):
        current_node = tree.get_node(index_split_node)

        leaf_node_value = self.sample_leaf_value(current_node.idx_data_points)

        new_leaf_node = LeafNode(index=index_split_node, value=leaf_node_value,
                                 idx_data_points=current_node.idx_data_points)
        tree.prune_tree(index_split_node, new_leaf_node)

    def is_variable_quantitative(self, index_variable):
        # TODO: implement check to find out if a variable in self.X is quantitative or qualitative
        return True

    def get_new_idx_data_points_aux(self, current_split_node):
        # TODO: remove.
        # This function works for quantitative and qualitative data
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

    def sample_leaf_value(self, idx_data_points):
        # Method extracted from the function assignLeafValsBySamplingFromPosteriorMeanAndSigsqAndUpdateYhats() of bartMachine
        current_num_observations = len(idx_data_points)
        responses = self.transformed_Y[idx_data_points]
        average_responses = responses / current_num_observations
        posterior_var = 1 / (1 / self.prior_sigma_square_mu + current_num_observations / self.current_sigma_square)
        posterior_mean = ((self.prior_mu_mu / self.prior_sigma_square_mu) +
                          (current_num_observations * average_responses / self.current_sigma_square)
                          ) * posterior_var

        # TODO: the samples of the normal can be cached for improved performance like bartpy does in the class NormalScalarSampler
        leaf_value = np.random.normal(posterior_mean, np.power(posterior_var, 0.5))
        return leaf_value

    def get_residual_tree(self, j):
        R_j = self.residuals_vector - self.trees[j].predict_output(self.num_observations)
        return R_j

    def draw_sigma_square_from_posterior(self):
        # Method extracted from the function drawSigsqFromPosterior() of bartMachine
        posterior_alpha = (self.prior_nu + self.num_observations) * 0.5
        quadratic_error = np.sum(np.square(self.transformed_Y - self.residuals_vector))
        posterior_beta = (self.prior_lambda * self.prior_nu + quadratic_error) * 0.5
        gamma_draw = np.random.gamma(posterior_alpha, posterior_beta)
        inverse_gamma_draw = 1 / gamma_draw
        return inverse_gamma_draw


def sample_from_discrete_uniform(lower, upper, size=None):
        samples = stats.randint.rvs(lower, upper, size=size)
        return samples


def compute_lambda_value_scaled_inverse_chi_square(overestimated_sigma, q, nu):
    # Method extracted from the function calculateHyperparameters() of bartMachine
    return stats.distributions.chi2.ppf(1 - q, nu) / nu * overestimated_sigma
