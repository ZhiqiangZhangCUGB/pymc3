from pymc3.bart.tree import SplitNode
from pymc3.bart.bart import BaseBART, ConjugateBART, BART
from pymc3.bart.exceptions import BARTParamsError
import pymc3 as pm
import numpy as np
import pytest


def create_X_corpus(number_elements_corpus, number_variates, X_min=0.0, X_max=1.0):
    return (X_max - X_min) * np.random.random_sample(size=(number_elements_corpus, number_variates)) + X_min


def create_Y_corpus(number_elements_corpus, Y_min=-0.5, Y_max=0.5):
    return (Y_max - Y_min) * np.random.random_sample(size=(number_elements_corpus,)) + Y_min


def test_correct_basebart_creation():
    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100)
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)

    assert base_bart.num_observations == 100
    assert base_bart.number_variates == 4
    assert base_bart.X is X
    assert base_bart.Y is Y
    assert base_bart.Y_transformed is Y
    assert len(base_bart.trees) == 200  # default value for number of trees
    assert np.array_equal(base_bart.trees[0][0].idx_data_points, np.array(range(100), dtype='int32'))

    X = create_X_corpus(number_elements_corpus=1000, number_variates=4, X_min=-300, X_max=500)
    Y = create_Y_corpus(number_elements_corpus=1000, Y_min=-50, Y_max=50)
    X[0:10, 2] = np.NaN
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y, m=50, alpha=0.9, beta=3.0, tree_sampler='ParticleGibbs', transform='regression')

    assert base_bart.num_observations == 1000
    assert base_bart.number_variates == 4
    assert base_bart.X is X
    assert base_bart.Y is Y
    assert base_bart.Y_transf_max_Y_transf_min_half_diff == 0.5
    assert len(base_bart.trees) == 50
    assert np.array_equal(base_bart.trees[0][0].idx_data_points, np.array(range(1000), dtype='int32'))


def test_incorrect_basebart_creation():
    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100)

    with pytest.raises(TypeError) as err:
        BaseBART(X=X, Y=Y)
    assert str(err.value) == "No model on context stack, which is needed to instantiate BART. Add variable inside a 'with model:' block."

    with pytest.raises(BARTParamsError) as err:
        vector = list(range(100))
        bad_X = [vector for _ in range(4)]
        with pm.Model():
            BaseBART(X=bad_X, Y=Y)
    assert str(err.value) == "The design matrix X type must be numpy.ndarray where every item type is numpy.float64"

    with pytest.raises(BARTParamsError) as err:
        vector = list(range(100))
        bad_X = [vector for _ in range(4)]
        bad_X = np.array(bad_X)  # invalid type of numpy array
        with pm.Model():
            BaseBART(X=bad_X, Y=Y)
    assert str(err.value) == "The design matrix X type must be numpy.ndarray where every item type is numpy.float64"

    with pytest.raises(BARTParamsError) as err:
        bad_X = np.array(list(range(100)), dtype='float64')
        with pm.Model():
            BaseBART(X=bad_X, Y=Y)
    assert str(err.value) == "The design matrix X must have two dimensions"

    with pytest.raises(BARTParamsError) as err:
        bad_Y = list(range(100))
        with pm.Model():
            BaseBART(X=X, Y=bad_Y)
    assert str(err.value) == "The response matrix Y type must be numpy.ndarray where every item type is numpy.float64"

    with pytest.raises(BARTParamsError) as err:
        bad_Y = np.array(list(range(100)))  # invalid type of numpy array
        with pm.Model():
            BaseBART(X=X, Y=bad_Y)
    assert str(err.value) == "The response matrix Y type must be numpy.ndarray where every item type is numpy.float64"

    with pytest.raises(BARTParamsError) as err:
        bad_Y = X
        with pm.Model():
            BaseBART(X=X, Y=bad_Y)
    assert str(err.value) == "The response matrix Y must have one dimension"

    with pytest.raises(BARTParamsError) as err:
        bad_Y = create_Y_corpus(number_elements_corpus=1000)
        with pm.Model():
            BaseBART(X=X, Y=bad_Y)
    assert str(err.value) == "The design matrix X and the response matrix Y must have the same number of elements"

    with pytest.raises(BARTParamsError) as err:
        bad_m = 50.0
        with pm.Model():
            BaseBART(X=X, Y=Y, m=bad_m)
    assert str(err.value) == "The number of trees m type must be int"

    with pytest.raises(BARTParamsError) as err:
        bad_m = 0
        with pm.Model():
            BaseBART(X=X, Y=Y, m=bad_m)
    assert str(err.value) == "The number of trees m must be greater than zero"

    with pytest.raises(BARTParamsError) as err:
        bad_alpha = 0
        with pm.Model():
            BaseBART(X=X, Y=Y, alpha=bad_alpha)
    assert str(err.value) == "The type for the alpha parameter for the tree structure must be float"

    with pytest.raises(BARTParamsError) as err:
        bad_alpha = 0.0
        with pm.Model():
            BaseBART(X=X, Y=Y, alpha=bad_alpha)
    assert str(err.value) == "The value for the alpha parameter for the tree structure must be in the interval (0, 1)"

    with pytest.raises(BARTParamsError) as err:
        bad_alpha = 1.0
        with pm.Model():
            BaseBART(X=X, Y=Y, alpha=bad_alpha)
    assert str(err.value) == "The value for the alpha parameter for the tree structure must be in the interval (0, 1)"

    with pytest.raises(BARTParamsError) as err:
        bad_beta = 30
        with pm.Model():
            BaseBART(X=X, Y=Y, beta=bad_beta)
    assert str(err.value) == "The type for the beta parameter for the tree structure must be float"

    with pytest.raises(BARTParamsError) as err:
        bad_beta = -1.0
        with pm.Model():
            BaseBART(X=X, Y=Y, beta=bad_beta)
    assert str(err.value) == 'The value for the beta parameter for the tree structure must be in the interval [0, float("inf"))'

    with pytest.raises(BARTParamsError) as err:
        bad_tree_sampler = 'bad_tree_sampler'
        with pm.Model():
            BaseBART(X=X, Y=Y, tree_sampler=bad_tree_sampler)
    assert str(err.value) == "{} is not a valid tree sampler".format(bad_tree_sampler)

    with pytest.raises(BARTParamsError) as err:
        bad_transform = 'bad_transform'
        with pm.Model():
            BaseBART(X=X, Y=Y, transform=bad_transform)
    assert str(err.value) == "{} is not a valid transformation for Y".format(bad_transform)


def test_correct_transform_Y():
    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100, Y_min=-130.0, Y_max=130.0)
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)
    assert base_bart.Y_transformed is base_bart.Y

    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y, transform='regression')
    assert base_bart.Y_transformed.min() == -0.5
    assert base_bart.Y_transformed.max() == 0.5

    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y, transform='classification')
    assert base_bart.Y_transformed.min() == -3.0
    assert base_bart.Y_transformed.max() == 3.0


def test_correct_un_transform_Y():
    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100, Y_min=-130.0, Y_max=130.0)
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)
    assert base_bart.un_transform_Y(base_bart.Y_transformed) is base_bart.Y

    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y, transform='regression')
    assert np.allclose(base_bart.un_transform_Y(base_bart.Y_transformed), base_bart.Y)

    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y, transform='classification')
    assert np.allclose(base_bart.un_transform_Y(base_bart.Y_transformed), base_bart.Y)

    Y_with_nan = Y.copy()
    Y_with_nan[0:10] = np.NaN
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y, transform='regression')
    un_transform_Y = base_bart.un_transform_Y(base_bart.Y_transformed)
    for i in range(base_bart.num_observations):
        if np.isnan(base_bart.Y[i]):
            assert np.isnan(un_transform_Y[i])
        else:
            assert np.isclose(un_transform_Y[i], base_bart.Y[i])


def test_correct_get_available_predictors():
    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100)
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)
    idx_data_points = np.array(range(base_bart.num_observations), dtype='int32')
    possible_splitting_variables = base_bart.get_available_predictors(idx_data_points)
    assert len(possible_splitting_variables) == 4

    X = np.ones_like(X)
    Y = create_Y_corpus(number_elements_corpus=100)
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)
    idx_data_points = np.array(range(base_bart.num_observations), dtype='int32')
    possible_splitting_variables = base_bart.get_available_predictors(idx_data_points)
    assert len(possible_splitting_variables) == 0

    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100)
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)
    idx_data_points = np.array([0], dtype='int32')
    possible_splitting_variables = base_bart.get_available_predictors(idx_data_points)
    assert len(possible_splitting_variables) == 0

    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100)
    X[:, 0] = 0.0
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)
    idx_data_points = np.array(range(base_bart.num_observations), dtype='int32')
    possible_splitting_variables = base_bart.get_available_predictors(idx_data_points)
    assert len(possible_splitting_variables) == 3


def test_correct_get_available_splitting_rules():
    X = np.array([[1.0, 2.0, 3.0, np.NaN], [2.0, 2.0, 3.0, 99.9], [3.0, 4.0, 3.0, -3.3]])
    Y = create_Y_corpus(number_elements_corpus=3)
    with pm.Model():
        base_bart = BaseBART(X=X, Y=Y)
    idx_split_variable = 0
    idx_data_points = np.array(range(base_bart.num_observations), dtype='int32')
    available_splitting_rules, _ = base_bart.get_available_splitting_rules(idx_data_points, idx_split_variable)
    assert len(available_splitting_rules) == 2
    assert np.array_equal(available_splitting_rules, np.array([1.0, 2.0]))

    idx_split_variable = 1
    idx_data_points = np.array(range(base_bart.num_observations), dtype='int32')
    available_splitting_rules, _ = base_bart.get_available_splitting_rules(idx_data_points, idx_split_variable)
    assert len(available_splitting_rules) == 1
    assert np.array_equal(available_splitting_rules, np.array([2.0]))

    idx_split_variable = 2
    idx_data_points = np.array(range(base_bart.num_observations), dtype='int32')
    available_splitting_rules, _ = base_bart.get_available_splitting_rules(idx_data_points, idx_split_variable)
    assert len(available_splitting_rules) == 0
    assert np.array_equal(available_splitting_rules, np.array([]))

    idx_split_variable = 3
    idx_data_points = np.array(range(base_bart.num_observations), dtype='int32')
    available_splitting_rules, _ = base_bart.get_available_splitting_rules(idx_data_points, idx_split_variable)
    assert len(available_splitting_rules) == 1
    assert np.array_equal(available_splitting_rules, np.array([-3.3]))


def test_correct_get_new_idx_data_points():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, np.NaN], [3.0, 4.0, 5.0], [4.0, 5.0, np.NaN]])
    Y = create_Y_corpus(number_elements_corpus=4)
    with pm.Model():
        conjugate_bart = ConjugateBART(X=X, Y=Y)

    split = SplitNode(index=0, idx_split_variable=0, split_value=2.0)
    idx_data_points = np.array(range(conjugate_bart.num_observations), dtype='int32')
    left_node_idx_data_points, right_node_idx_data_points = conjugate_bart.get_new_idx_data_points(split, idx_data_points)
    assert len(left_node_idx_data_points) == 2
    assert len(right_node_idx_data_points) == 2

    split = SplitNode(index=0, idx_split_variable=0, split_value=1.0)
    left_node_idx_data_points, right_node_idx_data_points = conjugate_bart.get_new_idx_data_points(split, idx_data_points)
    assert len(left_node_idx_data_points) == 1
    assert len(right_node_idx_data_points) == 3

    split = SplitNode(index=0, idx_split_variable=2, split_value=5.0)
    left_node_idx_data_points, right_node_idx_data_points = conjugate_bart.get_new_idx_data_points(split, idx_data_points)
    assert len(left_node_idx_data_points) == 2
    assert len(right_node_idx_data_points) == 2  # Here we found the two np.NaNs


def test_correct_successful_grow_tree():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    Y = create_Y_corpus(number_elements_corpus=4)
    with pm.Model():
        conjugate_bart = ConjugateBART(X=X, Y=Y)
    tree = conjugate_bart.trees[0]
    index_leaf_node = 0
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 2

    # grow again from the leaf node with more data points
    index_leaf_node = 1 if len(tree[2].idx_data_points) <= len(tree[1].idx_data_points) else 2
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 3


def test_correct_unsuccessful_grow_tree():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    Y = create_Y_corpus(number_elements_corpus=2)
    with pm.Model():
        conjugate_bart = ConjugateBART(X=X, Y=Y)
    tree = conjugate_bart.trees[0]
    index_leaf_node = 0
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 2

    # try to grow again
    index_leaf_node = 1
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert not successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 2

    X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    Y = create_Y_corpus(number_elements_corpus=2)
    with pm.Model():
        conjugate_bart = ConjugateBART(X=X, Y=Y)
    tree = conjugate_bart.trees[0]
    index_leaf_node = 0
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert not successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 1

    X = np.array([[1.0, 2.0, 1.0], [1.0, 4.0, 1.0], [1.0, np.NaN, 1.0]])
    Y = create_Y_corpus(number_elements_corpus=3)
    with pm.Model():
        conjugate_bart = ConjugateBART(X=X, Y=Y)
    tree = conjugate_bart.trees[0]
    index_leaf_node = 0
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 2

    index_leaf_node = 1
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert not successful_grow_tree  # Not enough data points
    assert len(tree.idx_leaf_nodes) == 2

    index_leaf_node = 2
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert not successful_grow_tree  # There are two data points but one of them has a np.NaN so it is not considered
    assert len(tree.idx_leaf_nodes) == 2


def test_correct_successful_prune_tree():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    Y = create_Y_corpus(number_elements_corpus=4)
    with pm.Model():
        conjugate_bart = ConjugateBART(X=X, Y=Y)
    tree = conjugate_bart.trees[0]
    index_leaf_node = 0
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 2

    index_split_node = 0
    conjugate_bart.prune_tree(tree, index_split_node)
    assert len(tree.idx_leaf_nodes) == 1

    X = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
    Y = create_Y_corpus(number_elements_corpus=4)
    with pm.Model():
        conjugate_bart = ConjugateBART(X=X, Y=Y)
    tree = conjugate_bart.trees[0]
    index_leaf_node = 0
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 2

    # grow again from the leaf node with more data points
    index_leaf_node = 1 if len(tree[2].idx_data_points) <= len(tree[1].idx_data_points) else 2
    successful_grow_tree = conjugate_bart.grow_tree(tree, index_leaf_node)
    assert successful_grow_tree
    assert len(tree.idx_leaf_nodes) == 3

    last_leaf_node = tree.idx_leaf_nodes[-1]
    index_split_node = tree[last_leaf_node].get_idx_parent_node()
    conjugate_bart.prune_tree(tree, index_split_node)
    assert len(tree.idx_leaf_nodes) == 2


def test_correct_conjugatebart_creation():
    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100)
    with pm.Model():
        ConjugateBART(X=X, Y=Y)


def test_incorrect_conjugatebart_creation():
    X = create_X_corpus(number_elements_corpus=100, number_variates=4)
    Y = create_Y_corpus(number_elements_corpus=100)

    with pytest.raises(BARTParamsError) as err:
        bad_nu = 3
        with pm.Model():
            ConjugateBART(X=X, Y=Y, nu=bad_nu)
    assert str(err.value) == "The type for the nu parameter related to the sigma prior must be float"

    with pytest.raises(BARTParamsError) as err:
        bad_nu = 2.0
        with pm.Model():
            ConjugateBART(X=X, Y=Y, nu=bad_nu)
    assert str(err.value) == "Chipman et al. discourage the use of nu less than 3.0"

    with pytest.raises(BARTParamsError) as err:
        bad_q = 2
        with pm.Model():
            ConjugateBART(X=X, Y=Y, q=bad_q)
    assert str(err.value) == "The type for the q parameter related to the sigma prior must be float"

    with pytest.raises(BARTParamsError) as err:
        bad_q = 0.0
        with pm.Model():
            ConjugateBART(X=X, Y=Y, q=bad_q)
    assert str(err.value) == "The value for the q parameter related to the sigma prior must be in the interval (0, 1)"

    with pytest.raises(BARTParamsError) as err:
        bad_q = 1.0
        with pm.Model():
            ConjugateBART(X=X, Y=Y, q=bad_q)
    assert str(err.value) == "The value for the q parameter related to the sigma prior must be in the interval (0, 1)"

    with pytest.raises(BARTParamsError) as err:
        bad_k = 1
        with pm.Model():
            ConjugateBART(X=X, Y=Y, k=bad_k)
    assert str(err.value) == "The type for the k parameter related to the mu_ij given T_j prior must be float"

    with pytest.raises(BARTParamsError) as err:
        bad_k = 0.0
        with pm.Model():
            ConjugateBART(X=X, Y=Y, k=bad_k)
    assert str(err.value) == 'The value for the k parameter k parameter related to the mu_ij given T_j prior must be in the interval (0, float("inf"))'
