from pymc3.bart.tree import SplitNode, LeafNode
from pymc3.bart.exceptions import TreeNodeError
import pytest
import numpy as np


def test_good_split_node_creation():
    s_quant = SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                        idx_data_points=np.array([1, 2, 3]))
    assert s_quant.index == 0
    assert np.array_equal(s_quant.idx_data_points, np.array([1, 2, 3]))
    assert s_quant.idx_split_variable == 2
    assert s_quant.type_split_variable == 'quantitative'
    assert s_quant.split_value == 3.0
    assert s_quant.depth == 0
    assert s_quant.operator == '<='

    s_qual = SplitNode(index=13, idx_split_variable=7, type_split_variable='qualitative', split_value={3, 2, 2},
                       idx_data_points=np.array([0, 1, 5, 7]))
    assert s_qual.index == 13
    assert np.array_equal(s_qual.idx_data_points, np.array([0, 1, 5, 7]))
    assert s_qual.idx_split_variable == 7
    assert s_qual.type_split_variable == 'qualitative'
    assert s_qual.split_value == {3, 2, 2}
    assert s_qual.depth == 3
    assert s_qual.operator == 'in'


def test_bad_split_nodes_creation():
    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=-1, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                  idx_data_points=np.array([1, 2, 3]))
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                  idx_data_points=np.array([2.3, 2.7, 4.56]))
    assert str(err.value) == 'Index of data points must be a numpy.ndarray of integers'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                  idx_data_points=np.array([], dtype=int))
    assert str(err.value) == 'Index of data points can not be empty'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=-2, type_split_variable='quantitative', split_value=3.0,
                  idx_data_points=np.array([1, 2, 3]))
    assert str(err.value) == 'Index of split variable must be a non-negative int'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='quant', split_value=3.0,
                  idx_data_points=np.array([1, 2, 3]))
    assert str(err.value) == 'Type of split variable must be "quantitative" or "qualitative"'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value='3',
                  idx_data_points=np.array([1, 2, 3]))
    assert str(err.value) == 'Node split value type must be float'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='qualitative', split_value=3.0,
                  idx_data_points=np.array([1, 2, 3]))
    assert str(err.value) == 'Node split value must be a set'


def test_good_leaf_node_creation():
    leaf_node = LeafNode(index=0, value=22.2, idx_data_points=np.array([1, 2, 3]))
    assert leaf_node.index == 0
    assert np.array_equal(leaf_node.idx_data_points, np.array([1, 2, 3]))
    assert leaf_node.value == 22.2

    leaf_node = LeafNode(index=0, value=22.2, idx_data_points=np.array([1, 6, 7]))
    assert leaf_node.index == 0
    assert np.array_equal(leaf_node.idx_data_points, np.array([1, 6, 7]))
    assert leaf_node.value == 22.2


def test_bad_leaf_nodes_creation():
    with pytest.raises(TreeNodeError) as err:
        LeafNode(index=-1, value=22.2, idx_data_points=np.array([1, 2, 3]))
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeNodeError) as err:
        LeafNode(index=0, value=22.2, idx_data_points=[1, 2, 3])
    assert str(err.value) == 'Index of data points must be a numpy.ndarray of integers'

    with pytest.raises(TreeNodeError) as err:
        LeafNode(index=0, value=22.2, idx_data_points=np.array([], dtype=int))
    assert str(err.value) == 'Index of data points can not be empty'

    with pytest.raises(TreeNodeError) as err:
        LeafNode(index=0, value='2', idx_data_points=np.array([1, 2, 3]))
    assert str(err.value) == 'Leaf node value type must be float'


def test_correct_evaluate_splitting_rule():
    quant_node = SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3,
                           idx_data_points=np.array([1, 2, 3]))

    x_quant_false = [0.0, 0.0, 5.5]
    assert quant_node.evaluate_splitting_rule(x_quant_false) is False
    x_quant_true = [0.0, 0.0, 2.2]
    assert quant_node.evaluate_splitting_rule(x_quant_true) is True
    x_quant_false_for_nan = [0.0, 0.0, np.NaN]
    assert quant_node.evaluate_splitting_rule(x_quant_false_for_nan) is False

    qual_node = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'C'},
                          idx_data_points=np.array([1, 2, 3]))

    x_qual_false = [0.0, 'B', 0.0]
    assert qual_node.evaluate_splitting_rule(x_qual_false) is False
    x_qual_true = [0.0, 'C', 0.0]
    assert qual_node.evaluate_splitting_rule(x_qual_true) is True
    x_qual_false_for_nan = [0.0, np.NaN, 0.0]
    assert qual_node.evaluate_splitting_rule(x_qual_false_for_nan) is False


def test_correct_get_idx_parent_node():
    node1 = SplitNode(index=13, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                      idx_data_points=np.array([1, 2, 3]))
    assert node1.get_idx_parent_node() == 6

    node2 = LeafNode(index=4, value=22.2, idx_data_points=np.array([1, 2, 3]))
    assert node2.get_idx_parent_node() == 1


def test_correct_get_idx_left_child():
    node1 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                      idx_data_points=np.array([1, 2, 3]))
    assert node1.get_idx_left_child() == 7

    node2 = LeafNode(index=4, value=22.2, idx_data_points=np.array([1, 2, 3]))
    assert node2.get_idx_left_child() == 9


def test_correct_get_idx_right_child():
    node1 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                      idx_data_points=np.array([1, 2, 3]))
    assert node1.get_idx_right_child() == 8

    node2 = LeafNode(index=4, value=22.2, idx_data_points=np.array([1, 2, 3]))
    assert node2.get_idx_right_child() == 10


def test_correct_is_left_child():
    node1 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                      idx_data_points=np.array([1, 2, 3]))
    assert node1.is_left_child() is True

    node2 = LeafNode(index=4, value=22.2, idx_data_points=np.array([1, 2, 3]))
    assert node2.is_left_child() is False


def test_correct_get_idx_sibling():
    node1 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                      idx_data_points=np.array([1, 2, 3]))
    assert node1.get_idx_sibling() == 4

    node2 = LeafNode(index=11, value=22.2, idx_data_points=np.array([1, 2, 3]))
    assert node2.get_idx_sibling() == 12


def test_correct_nodes_eq():
    split_node1 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                            idx_data_points=np.array([1, 2, 3]))
    split_node2 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                            idx_data_points=np.array([1, 2, 3]))
    split_node3 = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                            idx_data_points=np.array([1, 2, 3]))
    split_node4 = SplitNode(index=3, idx_split_variable=1, type_split_variable='quantitative', split_value=3.0,
                            idx_data_points=np.array([1, 2, 3]))
    split_node5 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=9.0,
                            idx_data_points=np.array([1, 2, 3]))
    split_node6 = SplitNode(index=3, idx_split_variable=2, type_split_variable='qualitative', split_value={1, 2},
                            idx_data_points=np.array([1, 2, 3]))
    split_node7 = SplitNode(index=3, idx_split_variable=2, type_split_variable='quantitative', split_value=3.0,
                            idx_data_points=np.array([1, 2]))
    assert (split_node1 == split_node2) is True
    assert (split_node1 == split_node3) is False
    assert (split_node1 == split_node4) is False
    assert (split_node1 == split_node5) is False
    assert (split_node1 == split_node6) is False
    assert (split_node1 == split_node7) is False

    leaf_node1 = LeafNode(index=1, value=22.2, idx_data_points=np.array([1, 2, 3]))
    leaf_node2 = LeafNode(index=1, value=22.2, idx_data_points=np.array([1, 2, 3]))
    leaf_node3 = LeafNode(index=2, value=22.2, idx_data_points=np.array([1, 2, 3]))
    leaf_node4 = LeafNode(index=1, value=55.5, idx_data_points=np.array([1, 2, 3]))
    leaf_node5 = LeafNode(index=1, value=55.5, idx_data_points=np.array([1, 2]))
    assert (leaf_node1 == leaf_node2) is True
    assert (leaf_node1 == leaf_node3) is False
    assert (leaf_node1 == leaf_node4) is False
    assert (leaf_node1 == leaf_node5) is False

    assert (split_node1 == leaf_node1) is False
