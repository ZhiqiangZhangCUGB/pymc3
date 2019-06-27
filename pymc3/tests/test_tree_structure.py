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
    assert str(err.value) == 'Node must have a parent node'

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
    assert t[1] == SplitNode(index=1, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)
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

    del t[3]
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
    assert str(err.value) == 'Invalid removal of node, leaving orphan children at index 1'


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

    assert t.traverse_tree(x_exit_node_1) is t[1]
    assert t.traverse_tree(x_exit_node_6) is t[6]
    assert t.traverse_tree(x_exit_node_11) is t[11]
    assert t.traverse_tree(x_exit_node_12) is t[12]


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
