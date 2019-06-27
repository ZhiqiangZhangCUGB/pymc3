from pymc3.bart.tree import SplitNode, LeafNode
from pymc3.bart.exceptions import TreeNodeError
import pytest


def test_good_split_node_creation():
    s_quant = SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value=3)
    assert s_quant.index == 0
    assert s_quant.idx_split_variable == 2
    assert s_quant.type_split_variable == 'quantitative'
    assert s_quant.split_value == 3
    assert s_quant.depth == 0
    assert s_quant.operator == '<='

    s_qual = SplitNode(index=13, idx_split_variable=7, type_split_variable='qualitative', split_value={3, 2, 2})
    assert s_qual.index == 13
    assert s_qual.idx_split_variable == 7
    assert s_qual.type_split_variable == 'qualitative'
    assert s_qual.split_value == {3, 2, 2}
    assert s_qual.depth == 3
    assert s_qual.operator == 'in'


def test_bad_split_nodes_creation():
    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=-1, idx_split_variable=2, type_split_variable='quantitative', split_value=3)
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=-2, type_split_variable='quantitative', split_value=3)
    assert str(err.value) == 'Index of split variable must be a non-negative int'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='quant', split_value=3)
    assert str(err.value) == 'Type of split variable must be "quantitative" or "qualitative"'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value='3')
    assert str(err.value) == 'Node split value must be a number'

    with pytest.raises(TreeNodeError) as err:
        SplitNode(index=0, idx_split_variable=2, type_split_variable='qualitative', split_value=3)
    assert str(err.value) == 'Node split value must be a set'


def test_good_leaf_node_creation():
    leaf_node = LeafNode(index=0, value=22.2)
    assert leaf_node.index == 0
    assert leaf_node.value == 22.2


def test_bad_leaf_nodes_creation():
    with pytest.raises(TreeNodeError) as err:
        LeafNode(index=-1, value=22.2)
    assert str(err.value) == 'Node index must be a non-negative int'

    with pytest.raises(TreeNodeError) as err:
        LeafNode(index=0, value='2')
    assert str(err.value) == 'Leaf node value must be float'


def test_correct_evaluate_splitting_rule():
    quant_node = SplitNode(index=0, idx_split_variable=2, type_split_variable='quantitative', split_value=2.3)

    x_quant_false = [0.0, 0.0, 5.5]
    assert quant_node.evaluate_splitting_rule(x_quant_false) is False
    x_quant_true = [0.0, 0.0, 2.2]
    assert quant_node.evaluate_splitting_rule(x_quant_true) is True

    qual_node = SplitNode(index=0, idx_split_variable=1, type_split_variable='qualitative', split_value={'A', 'C'})

    x_qual_false = [0.0, 'B', 0.0]
    assert qual_node.evaluate_splitting_rule(x_qual_false) is False
    x_qual_true = [0.0, 'C', 0.0]
    assert qual_node.evaluate_splitting_rule(x_qual_true) is True
