from pymc3.bart.tree import Tree, SplitNode, LeafNode
from pymc3.bart.exceptions import TreeStructureError
import numpy as np
import pytest


def test_correct_tree_structure_creation():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=2.3)
    t[1] = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    t[3] = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    t[4] = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))

    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [2, 3, 4]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=2.3),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')),
        3: LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32')),
        4: LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32')),
    }
    assert t.tree_structure == manual_tree

    t2 = Tree()
    t2.set_node(0, SplitNode(index=0, idx_split_variable=1, split_value=2.3))
    t2.set_node(1, LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32')))
    t2.set_node(2, LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')))

    assert t2.num_nodes == 3
    assert t2.idx_leaf_nodes == [1, 2]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=2.3),
        1: LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32')),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')),
    }
    assert t2.tree_structure == manual_tree


def test_incorrect_tree_structure_creation():
    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[2.0] = SplitNode(index=2, idx_split_variable=1, split_value=5.5)
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = [0, 1, 2]
    assert str(err.value) == 'Node class must be SplitNode or LeafNode'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = LeafNode(index=0, value=33.3, idx_data_points=np.array([1, 2, 3], dtype='int32'))
        t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    assert str(err.value) == 'Node index already exist in tree'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[5] = LeafNode(index=5, value=33.3, idx_data_points=np.array([1, 2, 3], dtype='int32'))
    assert str(err.value) == 'Root node must have index zero'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
        t[5] = LeafNode(index=5, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    assert str(err.value) == 'Invalid index, node must have a parent node'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = LeafNode(index=0, value=33.3, idx_data_points=np.array([1, 2, 3], dtype='int32'))
        t[1] = LeafNode(index=1, value=22.2, idx_data_points=np.array([1], dtype='int32'))
    assert str(err.value) == 'Parent node must be of class SplitNode'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = LeafNode(index=2, value=33.3, idx_data_points=np.array([1, 2, 3], dtype='int32'))
    assert str(err.value) == 'Node must have same index as tree index'


def test_correct_get_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    t[3] = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    t[4] = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))

    assert t[0] == SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    assert t.get_node(1) == SplitNode(index=1, idx_split_variable=2,
                                      split_value=2.3)
    assert t[2] == LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    assert t[3] == LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    assert t[4] == LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))


def test_incorrect_get_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32'))
    t[2] = LeafNode(index=2, value=22.2, idx_data_points=np.array([3], dtype='int32'))

    with pytest.raises(TreeStructureError) as err:
        node = t[-2]
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeStructureError) as err:
        node = t[4]
    assert str(err.value) == 'Node missing at index 4'


def test_correct_removal_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    t[3] = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    t[4] = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))

    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [2, 3, 4]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')),
        3: LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32')),
        4: LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32')),
    }
    assert t.tree_structure == manual_tree

    del t[4]
    assert t.num_nodes == 4
    assert t.idx_leaf_nodes == [2, 3]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')),
        3: LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32')),
    }
    assert t.tree_structure == manual_tree

    t.delete_node(3)
    assert t.num_nodes == 3
    assert t.idx_leaf_nodes == [2]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')),
    }
    assert t.tree_structure == manual_tree

    del t[1]
    assert t.num_nodes == 2
    assert t.idx_leaf_nodes == [2]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')),
    }
    assert t.tree_structure == manual_tree

    t[1] = LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32'))
    assert t.num_nodes == 3
    assert t.idx_leaf_nodes == [2, 1]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32')),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32')),
    }
    assert t.tree_structure == manual_tree


def test_incorrect_removal_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32'))
    t[2] = LeafNode(index=2, value=22.2, idx_data_points=np.array([3], dtype='int32'))

    with pytest.raises(TreeStructureError) as err:
        del t[1.0]
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeStructureError) as err:
        del t[4]
    assert str(err.value) == 'Node missing at index 4'

    with pytest.raises(TreeStructureError) as err:
        del t[0]
    assert str(err.value) == 'Invalid removal of node, leaving two orphans nodes'


def test_correct_traverse_tree():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = LeafNode(index=1, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    t[2] = SplitNode(index=2, idx_split_variable=2, split_value=2.3)
    t[5] = SplitNode(index=5, idx_split_variable=0, split_value=7.7)
    t[6] = LeafNode(index=6, value=44.4, idx_data_points=np.array([4, 5, 6, 7], dtype='int32'))
    t[11] = LeafNode(index=11, value=22.2, idx_data_points=np.array([5, 6, 7], dtype='int32'))
    t[12] = LeafNode(index=12, value=33.3, idx_data_points=np.array([6, 7], dtype='int32'))

    x_exit_node_1 = [0.0, 0.0, 0.0]
    x_exit_node_6 = [0.0, 10.0, 55.5]
    x_exit_node_6_nan = [0.0, 10.0, np.NaN]
    x_exit_node_6_nan_2 = [0.0, np.NaN, 55.5]
    x_exit_node_11 = [3.0, 10.0, 2.0]
    x_exit_node_12 = [9.0, 10.0, 2.0]
    x_exit_node_12_nan = [np.NaN, 10.0, 2.0]

    assert t._traverse_tree(x_exit_node_1) is t[1]
    assert t._traverse_tree(x_exit_node_6) is t[6]
    assert t._traverse_tree(x_exit_node_6_nan) is t[6]
    assert t._traverse_tree(x_exit_node_6_nan_2) is t[6]
    assert t._traverse_tree(x_exit_node_11) is t[11]
    assert t._traverse_tree(x_exit_node_12) is t[12]
    assert t._traverse_tree(x_exit_node_12_nan) is t[12]


def test_correct_out_of_sample_predict():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = LeafNode(index=1, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    t[2] = SplitNode(index=2, idx_split_variable=2, split_value=2.3)
    t[5] = SplitNode(index=5, idx_split_variable=0, split_value=7.7)
    t[6] = LeafNode(index=6, value=44.4, idx_data_points=np.array([4, 5, 6, 7], dtype='int32'))
    t[11] = LeafNode(index=11, value=22.2, idx_data_points=np.array([5, 6, 7], dtype='int32'))
    t[12] = LeafNode(index=12, value=33.3, idx_data_points=np.array([6, 7], dtype='int32'))

    x_exit_node_1 = [0.0, 0.0, 0.0]
    x_exit_node_6 = [0.0, 10.0, 55.5]
    x_exit_node_6_nan = [0.0, 10.0, np.NaN]
    x_exit_node_6_nan_2 = [0.0, np.NaN, 55.5]
    x_exit_node_11 = [3.0, 10.0, 2.0]
    x_exit_node_12 = [9.0, 10.0, 2.0]
    x_exit_node_12_nan = [np.NaN, 10.0, 2.0]

    assert t.out_of_sample_predict(x_exit_node_1) == 11.1
    assert t.out_of_sample_predict(x_exit_node_6) == 44.4
    assert t.out_of_sample_predict(x_exit_node_6_nan) == 44.4
    assert t.out_of_sample_predict(x_exit_node_6_nan_2) == 44.4
    assert t.out_of_sample_predict(x_exit_node_11) == 22.2
    assert t.out_of_sample_predict(x_exit_node_12) == 33.3
    assert t.out_of_sample_predict(x_exit_node_12_nan) == 33.3


def test_correct_grow_tree():
    t = Tree()
    t[0] = LeafNode(index=0, value=33.3, idx_data_points=np.array([1, 2, 3, 4], dtype='int32'))
    manual_tree = {
        0: LeafNode(index=0, value=33.3, idx_data_points=np.array([1, 2, 3, 4], dtype='int32')),
    }
    assert t.tree_structure == manual_tree
    assert t.num_nodes == 1
    assert t.idx_leaf_nodes == [0]
    assert t.idx_prunable_split_nodes == []

    new_split_node = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    new_left_node = LeafNode(index=1, value=33.3, idx_data_points=np.array([1, 2], dtype='int32'))
    new_right_node = LeafNode(index=2, value=33.3, idx_data_points=np.array([3, 4], dtype='int32'))
    t.grow_tree(0, new_split_node, new_left_node, new_right_node)
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: LeafNode(index=1, value=33.3, idx_data_points=np.array([1, 2], dtype='int32')),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3, 4], dtype='int32')),
    }
    assert t.tree_structure == manual_tree
    assert t.num_nodes == 3
    assert t.idx_leaf_nodes == [1, 2]
    assert t.idx_prunable_split_nodes == [0]

    new_split_node = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    new_left_node = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    new_right_node = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))
    t.grow_tree(1, new_split_node, new_left_node, new_right_node)
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: LeafNode(index=2, value=33.3, idx_data_points=np.array([3, 4], dtype='int32')),
        3: LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32')),
        4: LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))
    }
    assert t.tree_structure == manual_tree
    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [2, 3, 4]
    assert t.idx_prunable_split_nodes == [1]

    new_split_node = SplitNode(index=2, idx_split_variable=2, split_value=2.3)
    new_left_node = LeafNode(index=5, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    new_right_node = LeafNode(index=6, value=33.3, idx_data_points=np.array([4], dtype='int32'))
    t.grow_tree(2, new_split_node, new_left_node, new_right_node)
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: SplitNode(index=2, idx_split_variable=2, split_value=2.3),
        3: LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32')),
        4: LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32')),
        5: LeafNode(index=5, value=33.3, idx_data_points=np.array([3], dtype='int32')),
        6: LeafNode(index=6, value=33.3, idx_data_points=np.array([4], dtype='int32')),
    }
    assert t.tree_structure == manual_tree
    assert t.num_nodes == 7
    assert t.idx_leaf_nodes == [3, 4, 5, 6]
    assert t.idx_prunable_split_nodes == [1, 2]


def test_incorrect_grow_tree():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3, idx_data_points=np.array([4], dtype='int32'))
    t[3] = LeafNode(index=3, value=11.1, idx_data_points=np.array([1, 2], dtype='int32'))
    t[4] = LeafNode(index=4, value=22.2, idx_data_points=np.array([3], dtype='int32'))

    with pytest.raises(TreeStructureError) as err:
        new_split_node = SplitNode(index=1, idx_split_variable=0, split_value=100.0)
        new_left_node = LeafNode(index=3, value=666.6, idx_data_points=np.array([4], dtype='int32'))
        new_right_node = LeafNode(index=4, value=122.12, idx_data_points=np.array([5], dtype='int32'))
        t.grow_tree(1, new_split_node, new_left_node, new_right_node)
    assert str(err.value) == 'The tree grows from the leaves'

    with pytest.raises(TreeStructureError) as err:
        new_not_split_node = LeafNode(index=2, value=666.6, idx_data_points=np.array([4, 5], dtype='int32'))
        new_left_node = LeafNode(index=3, value=666.6, idx_data_points=np.array([4], dtype='int32'))
        new_right_node = LeafNode(index=4, value=122.12, idx_data_points=np.array([5], dtype='int32'))
        t.grow_tree(2, new_not_split_node, new_left_node, new_right_node)
    assert str(err.value) == 'The node that replaces the leaf node must be SplitNode'

    with pytest.raises(TreeStructureError) as err:
        new_split_node = SplitNode(index=2, idx_split_variable=0, split_value=100.0)
        new_left_node = LeafNode(index=3, value=666.6, idx_data_points=np.array([4], dtype='int32'))
        new_right_node = SplitNode(index=4, idx_split_variable=0, split_value=100.0)
        t.grow_tree(2, new_split_node, new_left_node, new_right_node)
    assert str(err.value) == 'The new leaves must be LeafNode'


def test_correct_prune_tree():
    t = Tree()
    t[0] = LeafNode(index=0, value=33.3, idx_data_points=np.array([1, 2, 3, 4], dtype='int32'))

    new_split_node = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    new_left_node = LeafNode(index=1, value=33.3, idx_data_points=np.array([1, 2], dtype='int32'))
    new_right_node = LeafNode(index=2, value=33.3, idx_data_points=np.array([3, 4], dtype='int32'))
    t.grow_tree(0, new_split_node, new_left_node, new_right_node)

    new_split_node = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    new_left_node = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    new_right_node = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))
    t.grow_tree(1, new_split_node, new_left_node, new_right_node)

    new_split_node = SplitNode(index=2, idx_split_variable=2, split_value=2.3)
    new_left_node = LeafNode(index=5, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    new_right_node = LeafNode(index=6, value=33.3, idx_data_points=np.array([4], dtype='int32'))
    t.grow_tree(2, new_split_node, new_left_node, new_right_node)

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: SplitNode(index=2, idx_split_variable=2, split_value=2.3),
        3: LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32')),
        4: LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32')),
        5: LeafNode(index=5, value=33.3, idx_data_points=np.array([3], dtype='int32')),
        6: LeafNode(index=6, value=33.3, idx_data_points=np.array([4], dtype='int32')),
    }
    assert t.tree_structure == manual_tree
    assert t.num_nodes == 7
    assert t.idx_leaf_nodes == [3, 4, 5, 6]
    assert t.idx_prunable_split_nodes == [1, 2]

    new_leaf_node = LeafNode(index=2, value=11.1, idx_data_points=np.array([3, 4], dtype='int32'))
    t.prune_tree(2, new_leaf_node)
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: SplitNode(index=1, idx_split_variable=2, split_value=2.3),
        2: LeafNode(index=2, value=11.1, idx_data_points=np.array([3, 4], dtype='int32')),
        3: LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32')),
        4: LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32')),
    }
    assert t.tree_structure == manual_tree
    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [3, 4, 2]
    assert t.idx_prunable_split_nodes == [1]

    new_leaf_node = LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32'))
    t.prune_tree(1, new_leaf_node)
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, split_value=5.5),
        1: LeafNode(index=1, value=11.1, idx_data_points=np.array([1, 2], dtype='int32')),
        2: LeafNode(index=2, value=11.1, idx_data_points=np.array([3, 4], dtype='int32')),
    }
    assert t.tree_structure == manual_tree
    assert t.num_nodes == 3
    assert t.idx_leaf_nodes == [2, 1]
    assert t.idx_prunable_split_nodes == [0]


def test_incorrect_prune_tree():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t[1] = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3, idx_data_points=np.array([4], dtype='int32'))
    t[3] = LeafNode(index=3, value=11.1, idx_data_points=np.array([1, 2], dtype='int32'))
    t[4] = LeafNode(index=4, value=22.2, idx_data_points=np.array([3], dtype='int32'))

    with pytest.raises(TreeStructureError) as err:
        new_leaf_node = LeafNode(index=2, value=666.6, idx_data_points=np.array([4], dtype='int32'))
        t.prune_tree(2, new_leaf_node)
    assert str(err.value) == 'Only SplitNodes are prunable'

    with pytest.raises(TreeStructureError) as err:
        new_leaf_node = LeafNode(index=0, value=666.6, idx_data_points=np.array([1, 2, 3, 4], dtype='int32'))
        t.prune_tree(0, new_leaf_node)
    assert str(err.value) == 'SplitNodes must have two LeafNodes as children to be prunable'


def test_correct_trees_eq():
    t1 = Tree()
    t1[0] = LeafNode(index=0, value=33.3, idx_data_points=np.array([1, 2, 3, 4], dtype='int32'))

    new_split_node = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    new_left_node = LeafNode(index=1, value=33.3, idx_data_points=np.array([1, 2], dtype='int32'))
    new_right_node = LeafNode(index=2, value=33.3, idx_data_points=np.array([3, 4], dtype='int32'))
    t1.grow_tree(0, new_split_node, new_left_node, new_right_node)

    new_split_node = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    new_left_node = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    new_right_node = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))
    t1.grow_tree(1, new_split_node, new_left_node, new_right_node)

    new_split_node = SplitNode(index=2, idx_split_variable=2, split_value=2.3)
    new_left_node = LeafNode(index=5, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    new_right_node = LeafNode(index=6, value=33.3, idx_data_points=np.array([4], dtype='int32'))
    t1.grow_tree(2, new_split_node, new_left_node, new_right_node)

    t2 = Tree()
    t2[0] = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    t2[1] = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    t2[2] = SplitNode(index=2, idx_split_variable=2, split_value=2.3)
    t2[3] = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    t2[4] = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))
    t2[5] = LeafNode(index=5, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    t2[6] = LeafNode(index=6, value=33.3, idx_data_points=np.array([4], dtype='int32'))
    t2.idx_prunable_split_nodes = [1, 2]

    t3 = Tree()
    t3[0] = LeafNode(index=0, value=33.3, idx_data_points=np.array([1, 2, 3, 4], dtype='int32'))

    new_split_node = SplitNode(index=0, idx_split_variable=1, split_value=5.5)
    new_left_node = LeafNode(index=1, value=33.3, idx_data_points=np.array([1, 2], dtype='int32'))
    new_right_node = LeafNode(index=2, value=33.3, idx_data_points=np.array([3, 4], dtype='int32'))
    t3.grow_tree(0, new_split_node, new_left_node, new_right_node)

    new_split_node = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    new_left_node = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    new_right_node = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))
    t3.grow_tree(1, new_split_node, new_left_node, new_right_node)

    assert (t1 == t2) is True
    assert (t1 == t3) is False


def test_correct_init_tree():
    t1 = Tree()
    t1[0] = LeafNode(index=0, value=33.3, idx_data_points=np.array([4, 5], dtype='int32'))

    t2 = Tree.init_tree(tree_id=0, leaf_node_value=33.3, idx_data_points=np.array([4, 5], dtype='int32'))

    assert t1 == t2


def test_correct_get_current_idx_data_points():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, split_value=2.3)
    t[1] = SplitNode(index=1, idx_split_variable=2, split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3, idx_data_points=np.array([3], dtype='int32'))
    t[3] = LeafNode(index=3, value=11.1, idx_data_points=np.array([1], dtype='int32'))
    t[4] = LeafNode(index=4, value=22.2, idx_data_points=np.array([2], dtype='int32'))

    assert np.array_equal(t.get_current_idx_data_points(index_node=0), np.array([1, 2, 3], dtype='int32'))
    assert np.array_equal(t.get_current_idx_data_points(index_node=1), np.array([1, 2], dtype='int32'))
    assert np.array_equal(t.get_current_idx_data_points(index_node=2), np.array([3], dtype='int32'))
    assert np.array_equal(t.get_current_idx_data_points(index_node=3), np.array([1], dtype='int32'))
    assert np.array_equal(t.get_current_idx_data_points(index_node=4), np.array([2], dtype='int32'))
