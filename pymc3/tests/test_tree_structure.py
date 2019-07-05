from pymc3.bart.tree import Tree, SplitNode, LeafNode
from pymc3.bart.exceptions import TreeStructureError
import pytest


def test_correct_tree_structure_creation():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3)
    t[3] = LeafNode(index=3, value=11.1)
    t[4] = LeafNode(index=4, value=22.2)

    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [2, 3, 4]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3),
        2: LeafNode(index=2, value=33.3),
        3: LeafNode(index=3, value=11.1),
        4: LeafNode(index=4, value=22.2),
    }
    assert t.tree_structure == manual_tree

    t2 = Tree()
    t2.set_node(0, SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}))
    t2.set_node(1, LeafNode(index=1, value=11.1))
    t2.set_node(2, LeafNode(index=2, value=33.3))

    assert t2.num_nodes == 3
    assert t2.idx_leaf_nodes == [1, 2]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: LeafNode(index=1, value=11.1),
        2: LeafNode(index=2, value=33.3),
    }
    assert t2.tree_structure == manual_tree


def test_incorrect_tree_structure_creation():
    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[2.0] = SplitNode(index=2, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = [0, 1, 2]
    assert str(err.value) == 'Node class must be SplitNode or LeafNode'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = LeafNode(index=0, value=33.3)
        t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    assert str(err.value) == 'Node index already exist in tree'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[5] = LeafNode(index=5, value=33.3)
    assert str(err.value) == 'Root node must have index zero'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
        t[5] = LeafNode(index=5, value=33.3)
    assert str(err.value) == 'Invalid index, node must have a parent node'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = LeafNode(index=0, value=33.3)
        t[1] = LeafNode(index=1, value=22.2)
    assert str(err.value) == 'Parent node must be of class SplitNode'

    with pytest.raises(TreeStructureError) as err:
        t = Tree()
        t[0] = LeafNode(index=2, value=33.3)
    assert str(err.value) == 'Node must have same index as tree index'


def test_correct_get_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3)
    t[3] = LeafNode(index=3, value=11.1)
    t[4] = LeafNode(index=4, value=22.2)

    assert t[0] == SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    assert t.get_node(1) == SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    assert t[2] == LeafNode(index=2, value=33.3)
    assert t[3] == LeafNode(index=3, value=11.1)
    assert t[4] == LeafNode(index=4, value=22.2)


def test_incorrect_get_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = LeafNode(index=1, value=11.1)
    t[2] = LeafNode(index=2, value=22.2)

    with pytest.raises(TreeStructureError) as err:
        node = t[-2]
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeStructureError) as err:
        node = t[4]
    assert str(err.value) == 'Node missing at index 4'


def test_correct_removal_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3)
    t[3] = LeafNode(index=3, value=11.1)
    t[4] = LeafNode(index=4, value=22.2)

    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [2, 3, 4]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3),
        2: LeafNode(index=2, value=33.3),
        3: LeafNode(index=3, value=11.1),
        4: LeafNode(index=4, value=22.2),
    }
    assert t.tree_structure == manual_tree

    del t[4]
    assert t.num_nodes == 4
    assert t.idx_leaf_nodes == [2, 3]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3),
        2: LeafNode(index=2, value=33.3),
        3: LeafNode(index=3, value=11.1),
    }
    assert t.tree_structure == manual_tree

    t.delete_node(3)
    assert t.num_nodes == 3
    assert t.idx_leaf_nodes == [2]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3),
        2: LeafNode(index=2, value=33.3),
    }
    assert t.tree_structure == manual_tree

    del t[1]
    assert t.num_nodes == 2
    assert t.idx_leaf_nodes == [2]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        2: LeafNode(index=2, value=33.3),
    }
    assert t.tree_structure == manual_tree

    t[1] = LeafNode(index=1, value=11.1)
    assert t.num_nodes == 3
    assert t.idx_leaf_nodes == [2, 1]
    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: LeafNode(index=1, value=11.1),
        2: LeafNode(index=2, value=33.3),
    }
    assert t.tree_structure == manual_tree


def test_incorrect_removal_node():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = LeafNode(index=1, value=11.1)
    t[2] = LeafNode(index=2, value=22.2)

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
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'D'})
    t[1] = LeafNode(index=1, value=11.1)
    t[2] = SplitNode(index=2, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[5] = SplitNode(index=5, idx_split_variable=0, type_split_variable='quantitative', split_value=7.7)
    t[6] = LeafNode(index=6, value=44.4)
    t[11] = LeafNode(index=11, value=22.2)
    t[12] = LeafNode(index=12, value=33.3)

    x_exit_node_1 = [0.0, 'D', 0.0]
    x_exit_node_6 = [0.0, 'B', 55.5]
    x_exit_node_11 = [3.0, 'B', 2.0]
    x_exit_node_12 = [9.0, 'B', 2.0]

    assert t._traverse_tree(x_exit_node_1) is t[1]
    assert t._traverse_tree(x_exit_node_6) is t[6]
    assert t._traverse_tree(x_exit_node_11) is t[11]
    assert t._traverse_tree(x_exit_node_12) is t[12]


def test_correct_out_of_sample_predict():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'D'})
    t[1] = LeafNode(index=1, value=11.1)
    t[2] = SplitNode(index=2, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[5] = SplitNode(index=5, idx_split_variable=0, type_split_variable='quantitative', split_value=7.7)
    t[6] = LeafNode(index=6, value=44.4)
    t[11] = LeafNode(index=11, value=22.2)
    t[12] = LeafNode(index=12, value=33.3)

    x_exit_node_1 = [0.0, 'D', 0.0]
    x_exit_node_6 = [0.0, 'B', 55.5]
    x_exit_node_11 = [3.0, 'B', 2.0]
    x_exit_node_12 = [9.0, 'B', 2.0]

    assert t.out_of_sample_predict(x_exit_node_1) == 11.1
    assert t.out_of_sample_predict(x_exit_node_6) == 44.4
    assert t.out_of_sample_predict(x_exit_node_11) == 22.2
    assert t.out_of_sample_predict(x_exit_node_12) == 33.3


def test_correct_get_idx_prunable_nodes_list():
    t1 = Tree()
    assert t1.get_idx_prunable_nodes_list() == []

    t2 = Tree()
    t2[0] = LeafNode(index=0, value=44.4)
    assert t2.get_idx_prunable_nodes_list() == []

    t3 = Tree()
    t3[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'D'})
    t3[1] = LeafNode(index=1, value=44.4)
    t3[2] = LeafNode(index=2, value=33.3)
    assert t3.get_idx_prunable_nodes_list() == [0]

    t4 = Tree()
    t4[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'D'})
    t4[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.2)
    t4[2] = SplitNode(index=2, idx_split_variable=3, type_split_variable='quantitative', split_value=3.3)
    t4[3] = SplitNode(index=3, idx_split_variable=4, type_split_variable='quantitative', split_value=4.4)
    t4[4] = SplitNode(index=4, idx_split_variable=5, type_split_variable='quantitative', split_value=5.5)
    t4[5] = SplitNode(index=5, idx_split_variable=6, type_split_variable='quantitative', split_value=6.6)
    t4[6] = LeafNode(index=6, value=99.9)
    t4[7] = LeafNode(index=7, value=11.1)
    t4[8] = LeafNode(index=8, value=22.2)
    t4[9] = LeafNode(index=9, value=33.3)
    t4[10] = LeafNode(index=10, value=44.4)
    t4[11] = LeafNode(index=11, value=55.5)
    t4[12] = LeafNode(index=12, value=66.6)
    assert t4.get_idx_prunable_nodes_list() == [3, 4, 5]


def test_correct_trees_eq():
    t1 = Tree()
    t1[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'D'})
    t1[1] = LeafNode(index=1, value=11.1)
    t1[2] = SplitNode(index=2, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t1[5] = SplitNode(index=5, idx_split_variable=0, type_split_variable='quantitative', split_value=7.7)
    t1[6] = LeafNode(index=6, value=44.4)
    t1[11] = LeafNode(index=11, value=22.2)
    t1[12] = LeafNode(index=12, value=33.3)

    t2 = Tree()
    t2[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'D'})
    t2[1] = LeafNode(index=1, value=11.1)
    t2[2] = SplitNode(index=2, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t2[5] = SplitNode(index=5, idx_split_variable=0, type_split_variable='quantitative', split_value=7.7)
    t2[6] = LeafNode(index=6, value=44.4)
    t2[11] = LeafNode(index=11, value=22.2)
    t2[12] = LeafNode(index=12, value=33.3)

    t3 = Tree()
    t3[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'D'})
    t3[1] = LeafNode(index=1, value=11.1)
    t3[2] = LeafNode(index=2, value=11.1)

    assert (t1 == t2) is True
    assert (t1 == t3) is False


def test_correct_grow_tree():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3)
    t[3] = LeafNode(index=3, value=11.1)
    t[4] = LeafNode(index=4, value=22.2)

    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [2, 3, 4]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3),
        2: LeafNode(index=2, value=33.3),
        3: LeafNode(index=3, value=11.1),
        4: LeafNode(index=4, value=22.2),
    }
    assert t.tree_structure == manual_tree

    new_split_node = SplitNode(index=2, idx_split_variable=0, type_split_variable='quantitative', split_value=100.0)
    new_left_node = LeafNode(index=5, value=666.6)
    new_right_node = LeafNode(index=6, value=122.12)
    t.grow_tree(2, new_split_node, new_left_node, new_right_node)

    assert t.num_nodes == 7
    assert t.idx_leaf_nodes == [3, 4, 5, 6]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3),
        2: SplitNode(index=2, idx_split_variable=0, type_split_variable='quantitative', split_value=100.0),
        3: LeafNode(index=3, value=11.1),
        4: LeafNode(index=4, value=22.2),
        5: LeafNode(index=5, value=666.6),
        6: LeafNode(index=6, value=122.12),
    }
    assert t.tree_structure == manual_tree


def test_incorrect_grow_tree():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3)
    t[3] = LeafNode(index=3, value=11.1)
    t[4] = LeafNode(index=4, value=22.2)

    with pytest.raises(TreeStructureError) as err:
        new_split_node = SplitNode(index=1, idx_split_variable=0, type_split_variable='quantitative', split_value=100.0)
        new_left_node = LeafNode(index=3, value=666.6)
        new_right_node = LeafNode(index=4, value=122.12)
        t.grow_tree(1, new_split_node, new_left_node, new_right_node)
    assert str(err.value) == 'The tree grows from the leaves'

    with pytest.raises(TreeStructureError) as err:
        new_not_split_node = LeafNode(index=2, value=666.6)
        new_left_node = LeafNode(index=3, value=666.6)
        new_right_node = LeafNode(index=4, value=122.12)
        t.grow_tree(2, new_not_split_node, new_left_node, new_right_node)
    assert str(err.value) == 'The node that replaces the leaf node must be SplitNode'

    with pytest.raises(TreeStructureError) as err:
        new_split_node = SplitNode(index=2, idx_split_variable=0, type_split_variable='quantitative', split_value=100.0)
        new_left_node = LeafNode(index=3, value=666.6)
        new_right_node = SplitNode(index=4, idx_split_variable=0, type_split_variable='quantitative', split_value=100.0)
        t.grow_tree(2, new_split_node, new_left_node, new_right_node)
    assert str(err.value) == 'The new leaves must be LeafNode'


def test_correct_prune_tree():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3)
    t[3] = LeafNode(index=3, value=11.1)
    t[4] = LeafNode(index=4, value=22.2)

    assert t.num_nodes == 5
    assert t.idx_leaf_nodes == [2, 3, 4]

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3),
        2: LeafNode(index=2, value=33.3),
        3: LeafNode(index=3, value=11.1),
        4: LeafNode(index=4, value=22.2),
    }
    assert t.tree_structure == manual_tree

    new_leaf_node = LeafNode(index=1, value=666.6)
    t.prune_tree(1, new_leaf_node)

    assert t.num_nodes == 3
    assert set(t.idx_leaf_nodes) == {1, 2}

    manual_tree = {
        0: SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3}),
        1: LeafNode(index=1, value=666.6),
        2: LeafNode(index=2, value=33.3),
    }
    assert t.tree_structure == manual_tree


def test_incorrect_prune_tree():
    t = Tree()
    t[0] = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={2, 2, 3})
    t[1] = SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
    t[2] = LeafNode(index=2, value=33.3)
    t[3] = LeafNode(index=3, value=11.1)
    t[4] = LeafNode(index=4, value=22.2)

    with pytest.raises(TreeStructureError) as err:
        new_leaf_node = LeafNode(index=2, value=666.6)
        t.prune_tree(2, new_leaf_node)
    assert str(err.value) == 'Only SplitNodes are prunable'
