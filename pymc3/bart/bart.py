from pymc3.bart.tree import Tree, SplitNode, LeafNode
from pymc3.model import Model
import numpy as np
from scipy import stats
from pymc3.bart.exceptions import (
    BARTParamsError,
)


class Particle:
    def __init__(self, tree):
        self.tree = tree.copy()  # Mantiene el arbol que nos interesa en este momento
        self.expansion_nodes = self.tree.idx_leaf_nodes.copy()  # This should be the array [0]
        self.tree_history = [self.tree.copy()]
        self.expansion_nodes_history = [self.expansion_nodes.copy()]
        self.weight = 1.0

    def sample_tree_sequential(self, bart):
        if self.expansion_nodes:
            index_leaf_node = self.expansion_nodes.pop(0)
            # Probability that this node will remain a leaf node
            log_prob = self.tree[index_leaf_node].prior_log_probability_node(bart.prior_alpha, bart.prior_beta)
            prob = np.exp(log_prob)
            should_grow = True if prob < np.random.random() else False
            if should_grow:
                grow_successful = bart.grow_tree(self.tree, index_leaf_node)
                # TODO: in case the grow_tree fails, should we try to sample the tree from another leaf node?
                if grow_successful:
                    # Add new leaf nodes indexes
                    new_indexes = self.tree.idx_leaf_nodes[-2:]
                    self.expansion_nodes.extend(new_indexes)
            self.tree_history.append(self.tree.copy())
            self.expansion_nodes_history.append(self.expansion_nodes.copy())

    def set_particle_to_step(self, t):
        if len(self.tree_history) <= t:
            self.tree = self.tree_history[-1]
            self.expansion_nodes = self.expansion_nodes_history[-1]
        else:
            self.tree = self.tree_history[t]
            self.expansion_nodes = self.expansion_nodes_history[t]

    def init_weight(self):
        # TODO
        return 1.

    def update_weight(self):
        # TODO
        pass


class TreeSampler:
    def step(self, bart, previous_tree):
        raise NotImplementedError()


class GrowPrune(TreeSampler):
    def __init__(self, cache_size=5000):
        self._disc_uniform_dist_sampler = DiscreteUniformDistributionSampler(cache_size)

    def step(self, bart, tree):
        # TODO: add calculation of likelihood and MH ratio to decide if we should change or not.
        # GROW:  1
        # PRUNE: 0
        step_done = False
        num_prunables_split_nodes = len(tree.idx_prunable_split_nodes)

        while not step_done:
            if bart.current_mcmc_step == 0 or num_prunables_split_nodes == 0:
                step_kind = 1
            else:
                step_kind = self._disc_uniform_dist_sampler.sample(0, 2)

            if step_kind:
                # GROW step
                num_leaf_nodes = len(tree.idx_leaf_nodes)
                idx = self._disc_uniform_dist_sampler.sample(0, num_leaf_nodes)
                index_leaf_node = tree.idx_leaf_nodes[idx]
                grow_successful = bart.grow_tree(tree, index_leaf_node)
                if grow_successful:
                    step_done = True
            else:
                # PRUNE step
                idx = self._disc_uniform_dist_sampler.sample(0, num_prunables_split_nodes)
                index_split_node = tree.idx_prunable_split_nodes[idx]
                bart.prune_tree(tree, index_split_node)
                step_done = True
        return tree


class ParticleGibbs(TreeSampler):
    def __init__(self, bart, num_particles=10, max_stages=5000):
        self.num_particles = num_particles
        self.max_stages = max_stages
        self.previous_trees_particles_list = []
        for i in range(bart.m):
            p = Particle(bart.trees[i])
            self.previous_trees_particles_list.append(p)

    def step(self, bart, tree):
        # Step 4 of algorithm
        R_j = bart.get_residuals(tree)
        list_of_particles = self.init_particles(tree.tree_id, R_j, bart.num_observations)

        # Step 5 of algorithm
        weights = np.array([p.init_weight() for p in list_of_particles])
        W = np.sum(weights)
        normalized_weights = weights / W

        for t in range(1, self.max_stages):
            # Step 7 of algorithm
            list_of_particles[0] = self.get_previous_tree_particle(tree.tree_id, t)

            for c in range(1, self.num_particles):  # This should be embarrassingly parallelizable
                # Step 9 of algorithm
                list_of_particles[c].sample_tree_sequential(bart)

            for c in range(self.num_particles):
                # Step 12 of algorithm
                list_of_particles[c].update_weight()

            weights = np.array([p.weight for p in list_of_particles])
            W = np.sum(weights)
            normalized_weights = weights / W

            # Step 15-16 of algorithm
            list_of_particles = self.resample(list_of_particles, normalized_weights)

            # Second part of step 16 of algorithm. TODO: Why they do this? It turns W into a scalar
            # weights = W / self.num_particles

            # Step 17 of algorithm
            non_available_nodes_for_expansion = True
            for c in range(self.num_particles):
                if len(list_of_particles[c].expansion_nodes) != 0:
                    non_available_nodes_for_expansion = False
                    break
            if non_available_nodes_for_expansion:
                break

        index = np.random.choice(self.num_particles, p=normalized_weights)
        self.previous_trees_particles_list[tree.tree_id] = list_of_particles[index]

        return list_of_particles[index].tree

    def get_previous_tree_particle(self, tree_id, t):
        previous_tree_particle = self.previous_trees_particles_list[tree_id]
        previous_tree_particle.set_particle_to_step(t)
        return previous_tree_particle

    def init_particles(self, tree_id, R_j, num_observations):
        list_of_particles = []
        initial_value_leaf_nodes = R_j.mean()
        initial_idx_data_points_leaf_nodes = np.array(range(num_observations), dtype='int32')
        new_tree = Tree.init_tree(tree_id=tree_id,
                                  leaf_node_value=initial_value_leaf_nodes,
                                  idx_data_points=initial_idx_data_points_leaf_nodes)
        for _ in range(self.num_particles):
            new_particle = Particle(new_tree)
            list_of_particles.append(new_particle)
        return list_of_particles

    def resample(self, list_of_particles, normalized_weights):
        # TODO
        return list_of_particles


class BaseBART:
    def __init__(self, X, Y, m=200, alpha=0.95, beta=2.0, tree_sampler='GrowPrune', transform=None, cache_size=5000):
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
            raise BARTParamsError('The response matrix Y type must be numpy.ndarray where every item'
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
        if tree_sampler not in ['GrowPrune', 'ParticleGibbs']:
            raise BARTParamsError('{} is not a valid tree sampler'.format(tree_sampler))
        if transform is not None and transform not in ['regression', 'classification']:
            raise BARTParamsError('{} is not a valid transformation for Y'.format(transform))

        self.X = X
        self.num_observations = X.shape[0]
        self.number_variates = X.shape[1]
        self.Y = Y

        self.transform = transform

        self.Y_min = self.Y.min()
        self.Y_max = self.Y.max()

        if self.transform == 'regression':
            self.Y_transf_max_Y_transf_min_half_diff = 0.5
        elif self.transform == 'classification':
            self.Y_transf_max_Y_transf_min_half_diff = 3.0
        elif self.transform is None:
            self.Y_transf_max_Y_transf_min_half_diff = (self.Y_max - self.Y_min) / 2

        self.Y_transformed = self.transform_Y(self.Y)
        self.overestimated_sigma = self.Y_transformed.std()

        self.m = m
        self.prior_alpha = alpha
        self.prior_beta = beta

        self._normal_dist_sampler = NormalDistributionSampler(cache_size)
        self._disc_uniform_dist_sampler = DiscreteUniformDistributionSampler(cache_size)

        self.trees = self.init_list_of_trees()

        self.tree_sampler = GrowPrune() if tree_sampler == 'GrowPrune' else ParticleGibbs(self)

        initial_value_leaf_nodes = self.Y_transformed.mean() / self.m
        # Diff trick to speed computation of residuals.
        # Taken from Section 3.1 of Kapelner, A and Bleich, J. bartMachine: A Powerful Tool for Machine Learning in R. ArXiv e-prints, 2013
        # The sum_trees_output will contain the sum of the predicted output for all trees.
        # When R_j is needed we subtract the current predicted output for tree T_j.
        self.sum_trees_output = np.ones_like(self.Y) * initial_value_leaf_nodes * self.m

    def init_list_of_trees(self):
        initial_value_leaf_nodes = self.Y_transformed.mean() / self.m
        initial_idx_data_points_leaf_nodes = np.array(range(self.num_observations), dtype='int32')
        list_of_trees = []
        for i in range(self.m):
            new_tree = Tree.init_tree(tree_id=i,
                                      leaf_node_value=initial_value_leaf_nodes,
                                      idx_data_points=initial_idx_data_points_leaf_nodes)
            list_of_trees.append(new_tree)
        return list_of_trees

    def __iter__(self):
        return iter(self.trees)

    def __repr__(self):
        raise NotImplementedError

    def transform_Y(self, Y):
        '''
        Transforms the output variable Y using the Min-Max Feature scaling normalization.
        The obtained range of Y is [-self.Y_transf_max_Y_transf_min_half_diff, self.Y_transf_max_Y_transf_min_half_diff]

        Parameters
        ----------
        Y

        Returns
        -------

        '''
        if self.transform:
            return (Y - self.Y_min) / (self.Y_max - self.Y_min) * (self.Y_transf_max_Y_transf_min_half_diff * 2)\
                   - self.Y_transf_max_Y_transf_min_half_diff
        else:
            return Y

    def un_transform_Y(self, Y):
        if self.transform:
            return (Y + self.Y_transf_max_Y_transf_min_half_diff) * (self.Y_max - self.Y_min)\
               / (self.Y_transf_max_Y_transf_min_half_diff * 2) + self.Y_min
        else:
            return Y

    def prediction_untransformed(self, x):
        sum_of_trees = 0.0
        for t in self.trees:
            sum_of_trees += t.out_of_sample_predict(x=x)
        return self.un_transform_Y(sum_of_trees)

    def sample_dist_splitting_variable(self, value):
        return self._disc_uniform_dist_sampler.sample(0, value)

    def sample_dist_splitting_rule_assignment(self, value):
        return self._disc_uniform_dist_sampler.sample(0, value)

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
        # The last value is not consider since if we choose it as the value of
        # the splitting rule assignment, it would leave the right subtree empty.
        return values[:-1], indices[:-1]

    def grow_tree(self, tree, index_leaf_node):
        # This can be unsuccessful when there are not available predictors
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
                                   split_value=selected_splitting_rule)

        left_node_idx_data_points, right_node_idx_data_points = self.get_new_idx_data_points(new_split_node, current_node.idx_data_points)

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
        # This is always successful because we call this method knowing the prunable split node to prune
        current_node = tree.get_node(index_split_node)
        idx_data_points = tree.get_current_idx_data_points(index_split_node)

        leaf_node_value = self.draw_leaf_value(tree, idx_data_points)

        new_leaf_node = LeafNode(index=index_split_node, value=leaf_node_value,
                                 idx_data_points=idx_data_points)
        tree.prune_tree(index_split_node, new_leaf_node)

    def get_new_idx_data_points(self, current_split_node, idx_data_points):
        idx_split_variable = current_split_node.idx_split_variable
        split_value = current_split_node.split_value

        left_idx = np.nonzero(self.X[idx_data_points, idx_split_variable] <= split_value)
        left_node_idx_data_points = idx_data_points[left_idx]
        right_idx = np.nonzero(~(self.X[idx_data_points, idx_split_variable] <= split_value))
        right_node_idx_data_points = idx_data_points[right_idx]

        return left_node_idx_data_points, right_node_idx_data_points

    def get_residuals(self, tree):
        R_j = self.Y_transformed - (self.sum_trees_output - tree.predict_output(self.num_observations))
        return R_j

    def draw_leaf_value(self, tree, idx_data_points):
        raise NotImplementedError


class BART(BaseBART):
    def __init__(self, X, Y, m=200, alpha=0.95, beta=2.0,
                 tree_sampler='GrowPrune', transform=None):
        super().__init__(X, Y, m, alpha, beta, tree_sampler, transform)

    def __repr__(self):
        representation = '''
BART(
     X = {},
     Y = {},
     m = {},
     alpha = {},
     beta = {},
     tree_sampler={!r},
     transform={!r})'''
        return representation.format(type(self.X), type(self.Y), self.m, self.prior_alpha, self.prior_beta,
                                     self.tree_sampler, self.transform)

    def draw_leaf_value(self, tree, idx_data_points):
        current_num_observations = len(idx_data_points)
        R_j = self.get_residuals(tree)
        node_responses = R_j[idx_data_points]
        node_average_responses = np.sum(node_responses) / current_num_observations

        posterior_mean = node_average_responses
        posterior_variance = node_responses.var()

        draw = posterior_mean + (self._normal_dist_sampler.sample() * np.power(posterior_variance, 0.5))
        return draw


class ConjugateBART(BaseBART):
    def __init__(self, X, Y, m=200, alpha=0.95, beta=2.0,
                 nu=3.0,
                 q=0.9,
                 k=2.0,
                 tree_sampler='GrowPrune',
                 num_mcmc_iterations=5000,
                 num_mcmc_burn_in_iterations=1000,
                 transform=None):

        super().__init__(X, Y, m, alpha, beta, tree_sampler, transform)
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

        self.prior_k = k
        self.prior_nu = nu
        self.prior_q = q
        self.prior_lambda = self.compute_lambda_value_scaled_inverse_chi_square(self.overestimated_sigma,
                                                                                self.prior_q, self.prior_nu)
        self.current_sigma = 1.0

        self.prior_mu_mu = 0.0
        self.prior_sigma_mu = self.Y_transf_max_Y_transf_min_half_diff / (self.prior_k * np.sqrt(self.m))

        self.current_mcmc_step = 0
        self.num_mcmc_iterations = num_mcmc_iterations
        self.num_mcmc_burn_in_iterations = num_mcmc_burn_in_iterations

        self.trace = {
            'trees': [],
            'yhats': [],
            'log_likelihood': [],
            'prior': [],
            'posterior': [],
        }
        self.trace['trees'].append(self.trees)
        self.trace['yhats'].append(self.sum_trees_output.copy())

    def __repr__(self):
        representation = '''
ConjugateBART(
     X = {},
     Y = {},
     m = {},
     alpha = {},
     beta = {},
     nu = {},
     q = {},
     k = {},
     tree_sampler={!r},
     transform={!r})'''
        return representation.format(type(self.X), type(self.Y), self.m, self.prior_alpha, self.prior_beta,
                                     self.prior_nu, self.prior_q, self.prior_k, self.tree_sampler, self.transform)

    def show_evolution_tree(self, tree_idx):
        try:
            import graphviz
        except ImportError:
            raise ImportError('This function requires the python library graphviz, along with binaries. '
                              'The easiest way to install all of this is by running\n\n'
                              '\tconda install -c conda-forge python-graphviz')
        g = graphviz.Digraph('tree_evolution')
        # The subgraph name needs to begin with 'cluster' (all lowercase)
        # so that Graphviz recognizes it as a special cluster subgraph
        for i in range(self.num_mcmc_iterations):
            cluster_name = 'cluster_{}'.format(i)
            graph = self.trace['trees'][i][tree_idx].make_digraph(cluster_name)
            g.subgraph(graph=graph)
        return g

    def sample(self):
        for i in range(1, self.num_mcmc_iterations):
            previous_bart_trees = self.trace['trees'][i - 1]

            current_bart_trees, yhats = self.do_one_mcmc_step(previous_bart_trees)
            self.current_mcmc_step = i

            self.trace['trees'].append(current_bart_trees)
            self.trace['yhats'].append(yhats)

        self.trees = self.trace['trees'][-1]

        trace_without_burn_in = {
            'trees': self.trace['trees'][self.num_mcmc_burn_in_iterations:],
            'yhats': self.trace['yhats'][self.num_mcmc_burn_in_iterations:],
        }
        return trace_without_burn_in

    def do_one_mcmc_step(self, previous_bart_trees):
        self.current_sigma = self.draw_sigma_from_posterior()

        current_bart_trees = []
        for previous_tree in previous_bart_trees:
            new_tree = self.sample_tree(previous_tree=previous_tree)
            current_bart_trees.append(new_tree)
            self.sum_trees_output -= previous_tree.predict_output(self.num_observations)
            self.sum_trees_output += new_tree.predict_output(self.num_observations)

        # return things related to the trace
        return current_bart_trees, self.sum_trees_output.copy()

    def sample_tree(self, previous_tree):
        tree = previous_tree.copy()
        # Se encarga de samplear la estructura del arbol y el valor del nodo hoja
        return self.tree_sampler.step(self, tree)

    def draw_leaf_value(self, tree, idx_data_points):
        # Method extracted from the function LeafNodeSampler.sample() of bartpy
        current_num_observations = len(idx_data_points)
        R_j = self.get_residuals(tree)
        node_responses = R_j[idx_data_points]
        node_average_responses = np.sum(node_responses) / current_num_observations

        prior_var = self.prior_sigma_mu ** 2
        likelihood_var = (self.current_sigma ** 2) / current_num_observations
        likelihood_mean = node_average_responses
        posterior_variance = 1. / (1. / prior_var + 1. / likelihood_var)
        posterior_mean = likelihood_mean * (prior_var / (likelihood_var + prior_var))
        draw = posterior_mean + (self._normal_dist_sampler.sample() * np.power(posterior_variance / self.m, 0.5))
        return draw

    def draw_sigma_from_posterior(self):
        # Method extracted from the function SigmaSampler.sample() of bartpy
        posterior_alpha = self.prior_nu + (self.num_observations / 2.)
        quadratic_error = np.sum(np.square(self.Y_transformed - self.sum_trees_output))
        posterior_beta = self.prior_lambda + (0.5 * quadratic_error)
        draw = np.power(np.random.gamma(posterior_alpha, 1. / posterior_beta), -0.5)
        return draw

    def one_mcmc_step_variable_importance(self, current_step):
        num_repetitions_variables = np.zeros(self.number_variates, dtype='int64')

        bart_trees = self.trace['trees'][current_step]
        for t in bart_trees:
            for node in t:
                if isinstance(node, SplitNode):
                    idx = node.idx_split_variable
                    num_repetitions_variables[idx] += 1
        total = num_repetitions_variables.sum()
        return num_repetitions_variables / total if total != 0 else num_repetitions_variables

    def variable_importance(self):
        number_mcmc_steps = self.num_mcmc_iterations - self.num_mcmc_burn_in_iterations
        proportion_repetitions_variables_all_steps = np.zeros((number_mcmc_steps, self.number_variates), dtype='int64')
        for i in range(self.num_mcmc_burn_in_iterations, self.num_mcmc_iterations):
            proportion_current_step = self.one_mcmc_step_variable_importance(current_step=i)
            idx = i - self.num_mcmc_burn_in_iterations
            proportion_repetitions_variables_all_steps[idx] = proportion_current_step
        return proportion_repetitions_variables_all_steps.sum(axis=0) / number_mcmc_steps

    @staticmethod
    def compute_lambda_value_scaled_inverse_chi_square(overestimated_sigma, q, nu):
        # Method extracted from the function calculateHyperparameters() of bartMachine
        return stats.distributions.chi2.ppf(1 - q, nu) / nu * overestimated_sigma


class NormalDistributionSampler:
    def __init__(self,
                 cache_size=1000):
        self._cache_size = cache_size
        self._cache = []

    def sample(self):
        if len(self._cache) == 0:
            self.refresh_cache()
        return self._cache.pop()

    def refresh_cache(self):
        self._cache = list(np.random.normal(size=self._cache_size))


class DiscreteUniformDistributionSampler:
    '''
    Draw samples from a discrete uniform distribution.
    Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high).
    '''
    def __init__(self,
                 cache_size=1000):
        self._cache_size = cache_size
        self._cache = []

    def sample(self, lower_limit, upper_limit):
        if len(self._cache) == 0:
            self.refresh_cache()
        return int(lower_limit + (upper_limit - lower_limit) * self._cache.pop())

    def refresh_cache(self):
        self._cache = list(np.random.random(size=self._cache_size))
