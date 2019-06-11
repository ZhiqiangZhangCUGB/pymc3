from pymc3.bart.tree import Node, build
from pymc3.bart.exceptions import (
    NodeValueError,
    NodeIndexError,
    NodeTypeError,
    NodeModifyError,
    NodeNotFoundError,
    NodeReferenceError,
)
import pytest


def test_node_set_attributes():
    root = Node(1)
    assert root.left is None
    assert root.right is None
    assert root.value == 1
    assert repr(root) == 'Node(1)'

    left_child = Node(2)
    root.left = left_child
    assert root.left is left_child
    assert root.right is None
    assert root.value == 1
    assert root.left.left is None
    assert root.left.right is None
    assert root.left.value == 2
    assert repr(left_child) == 'Node(2)'

    right_child = Node(3)
    root.right = right_child
    assert root.left is left_child
    assert root.right is right_child
    assert root.value == 1
    assert root.right.left is None
    assert root.right.right is None
    assert root.right.value == 3
    assert repr(right_child) == 'Node(3)'

    last_node = Node(4)
    left_child.right = last_node
    assert root.left.right is last_node
    assert root.left.right.value == 4
    assert repr(last_node) == 'Node(4)'

    with pytest.raises(NodeValueError) as err:
        # noinspection PyTypeChecker
        Node('this_is_not_an_integer')
    assert str(err.value) == 'node value must be a number'

    with pytest.raises(NodeTypeError) as err:
        # noinspection PyTypeChecker
        Node(1, 'this_is_not_a_node')
    assert str(err.value) == 'left child must be a Node instance'

    with pytest.raises(NodeTypeError) as err:
        # noinspection PyTypeChecker
        Node(1, Node(1), 'this_is_not_a_node')
    assert str(err.value) == 'right child must be a Node instance'

    with pytest.raises(NodeValueError) as err:
        root.value = 'this_is_not_an_integer'
    assert root.value == 1
    assert str(err.value) == 'node value must be a number'

    with pytest.raises(NodeTypeError) as err:
        root.left = 'this_is_not_a_node'
    assert root.left is left_child
    assert str(err.value) == 'left child must be a Node instance'

    with pytest.raises(NodeTypeError) as err:
        root.right = 'this_is_not_a_node'
    assert root.right is right_child
    assert str(err.value) == 'right child must be a Node instance'


def test_tree_build():
    root = build([])
    assert root is None

    root = build([1])
    assert root.value == 1
    assert root.left is None
    assert root.right is None

    root = build([1, 2])
    assert root.value == 1
    assert root.left.value == 2
    assert root.right is None

    root = build([1, 2, 3])
    assert root.value == 1
    assert root.left.value == 2
    assert root.right.value == 3
    assert root.left.left is None
    assert root.left.right is None
    assert root.right.left is None
    assert root.right.right is None

    root = build([1, 2, 3, None, 4])
    assert root.value == 1
    assert root.left.value == 2
    assert root.right.value == 3
    assert root.left.left is None
    assert root.left.right.value == 4
    assert root.right.left is None
    assert root.right.right is None
    assert root.left.right.left is None
    assert root.left.right.right is None

    with pytest.raises(NodeNotFoundError) as err:
        build([None, 1, 2])
    assert str(err.value) == 'parent node missing at index 0'

    with pytest.raises(NodeNotFoundError) as err:
        build([1, None, 2, 3, 4])
    assert str(err.value) == 'parent node missing at index 1'


def test_tree_get_node():
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.left.right.left = Node(6)

    assert root[0] is root
    assert root[1] is root.left
    assert root[2] is root.right
    assert root[3] is root.left.left
    assert root[4] is root.left.right
    assert root[9] is root.left.right.left

    for index in [5, 6, 7, 8, 10]:
        with pytest.raises(NodeNotFoundError) as err:
            assert root[index]
        assert str(err.value) == 'node missing at index {}'.format(index)

    with pytest.raises(NodeIndexError) as err:
        assert root[-1]
    assert str(err.value) == 'node index must be a non-negative int'


def test_tree_set_node():
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.left.right.left = Node(6)

    new_node_1 = Node(7)
    new_node_2 = Node(8)
    new_node_3 = Node(9)

    with pytest.raises(NodeModifyError) as err:
        root[0] = new_node_1
    assert str(err.value) == 'cannot modify the root node'

    with pytest.raises(NodeIndexError) as err:
        root[-1] = new_node_1
    assert str(err.value) == 'node index must be a non-negative int'

    with pytest.raises(NodeNotFoundError) as err:
        root[100] = new_node_1
    assert str(err.value) == 'parent node missing at index 49'

    root[10] = new_node_1
    assert root.value == 1
    assert root.left.value == 2
    assert root.right.value == 3
    assert root.left.left.value == 4
    assert root.left.right.value == 5
    assert root.left.right.left.value == 6
    assert root.left.right.right is new_node_1

    root[4] = new_node_2
    assert root.value == 1
    assert root.left.value == 2
    assert root.right.value == 3
    assert root.left.left.value == 4
    assert root.left.right.value == 8
    assert root.left.right.left is None
    assert root.left.right.right is None

    root[1] = new_node_3
    root[2] = new_node_2
    assert root.left is new_node_3
    assert root.right is new_node_2


def test_tree_del_node():
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.left.right.left = Node(6)

    with pytest.raises(NodeModifyError) as err:
        del root[0]
    assert str(err.value) == 'cannot delete the root node'

    with pytest.raises(NodeIndexError) as err:
        del root[-1]
    assert str(err.value) == 'node index must be a non-negative int'

    with pytest.raises(NodeNotFoundError) as err:
        del root[10]
    assert str(err.value) == 'no node to delete at index 10'

    with pytest.raises(NodeNotFoundError) as err:
        del root[100]
    assert str(err.value) == 'no node to delete at index 100'

    del root[3]
    assert root.left.left is None
    assert root.left.value == 2
    assert root.left.right.value == 5
    assert root.left.right.right is None
    assert root.left.right.left.value == 6
    assert root.left.right.left.left is None
    assert root.left.right.left.right is None
    assert root.right.value == 3
    assert root.right.left is None
    assert root.right.right is None
    assert root.size == 5

    del root[2]
    assert root.left.left is None
    assert root.left.value == 2
    assert root.left.right.value == 5
    assert root.left.right.right is None
    assert root.left.right.left.value == 6
    assert root.left.right.left.left is None
    assert root.left.right.left.right is None
    assert root.right is None
    assert root.size == 4

    del root[4]
    assert root.left.left is None
    assert root.left.right is None
    assert root.right is None
    assert root.size == 2

    del root[1]
    assert root.left is None
    assert root.right is None
    assert root.size == 1


def test_tree_properties():
    root = Node(1)
    assert root.properties == {
        'height': 0,
        'leaf_count': 1,
        'max_leaf_depth': 0,
        'max_node_value': 1,
        'min_leaf_depth': 0,
        'min_node_value': 1,
        'size': 1
    }
    assert root.height == 0
    assert root.leaf_count == 1
    assert root.max_leaf_depth == 0
    assert root.max_node_value == 1
    assert root.min_leaf_depth == 0
    assert root.min_node_value == 1
    assert root.size == len(root) == 1

    root.left = Node(2)
    assert root.properties == {
        'height': 1,
        'leaf_count': 1,
        'max_leaf_depth': 1,
        'max_node_value': 2,
        'min_leaf_depth': 1,
        'min_node_value': 1,
        'size': 2
    }
    assert root.height == 1
    assert root.leaf_count == 1
    assert root.max_leaf_depth == 1
    assert root.max_node_value == 2
    assert root.min_leaf_depth == 1
    assert root.min_node_value == 1
    assert root.size == len(root) == 2

    root.right = Node(3)
    assert root.properties == {
        'height': 1,
        'leaf_count': 2,
        'max_leaf_depth': 1,
        'max_node_value': 3,
        'min_leaf_depth': 1,
        'min_node_value': 1,
        'size': 3
    }
    assert root.height == 1
    assert root.leaf_count == 2
    assert root.max_leaf_depth == 1
    assert root.max_node_value == 3
    assert root.min_leaf_depth == 1
    assert root.min_node_value == 1
    assert root.size == len(root) == 3

    root.left.left = Node(4)
    assert root.properties == {
        'height': 2,
        'leaf_count': 2,
        'max_leaf_depth': 2,
        'max_node_value': 4,
        'min_leaf_depth': 1,
        'min_node_value': 1,
        'size': 4
    }
    assert root.height == 2
    assert root.leaf_count == 2
    assert root.max_leaf_depth == 2
    assert root.max_node_value == 4
    assert root.min_leaf_depth == 1
    assert root.min_node_value == 1
    assert root.size == len(root) == 4

    root.right.left = Node(5)
    assert root.properties == {
        'height': 2,
        'leaf_count': 2,
        'max_leaf_depth': 2,
        'max_node_value': 5,
        'min_leaf_depth': 2,
        'min_node_value': 1,
        'size': 5
    }
    assert root.height == 2
    assert root.leaf_count == 2
    assert root.max_leaf_depth == 2
    assert root.max_node_value == 5
    assert root.min_leaf_depth == 2
    assert root.min_node_value == 1
    assert root.size == len(root) == 5

    root.right.left.left = Node(6)
    assert root.properties == {
        'height': 3,
        'leaf_count': 2,
        'max_leaf_depth': 3,
        'max_node_value': 6,
        'min_leaf_depth': 2,
        'min_node_value': 1,
        'size': 6
    }
    assert root.height == 3
    assert root.leaf_count == 2
    assert root.max_leaf_depth == 3
    assert root.max_node_value == 6
    assert root.min_leaf_depth == 2
    assert root.min_node_value == 1
    assert root.size == len(root) == 6

    root.left.left.left = Node(7)
    assert root.properties == {
        'height': 3,
        'leaf_count': 2,
        'max_leaf_depth': 3,
        'max_node_value': 7,
        'min_leaf_depth': 3,
        'min_node_value': 1,
        'size': 7
    }
    assert root.height == 3
    assert root.leaf_count == 2
    assert root.max_leaf_depth == 3
    assert root.max_node_value == 7
    assert root.min_leaf_depth == 3
    assert root.min_node_value == 1
    assert root.size == len(root) == 7


def test_tree_traversal():
    n1 = Node(1)
    assert n1.levels == [[n1]]
    assert n1.leaves == [n1]
    assert n1.inorder == [n1]
    assert n1.preorder == [n1]
    assert n1.postorder == [n1]
    assert n1.levelorder == [n1]

    n2 = Node(2)
    n1.left = n2
    assert n1.levels == [[n1], [n2]]
    assert n1.leaves == [n2]
    assert n1.inorder == [n2, n1]
    assert n1.preorder == [n1, n2]
    assert n1.postorder == [n2, n1]
    assert n1.levelorder == [n1, n2]

    n3 = Node(3)
    n1.right = n3
    assert n1.levels == [[n1], [n2, n3]]
    assert n1.leaves == [n2, n3]
    assert n1.inorder == [n2, n1, n3]
    assert n1.preorder == [n1, n2, n3]
    assert n1.postorder == [n2, n3, n1]
    assert n1.levelorder == [n1, n2, n3]

    n4 = Node(4)
    n5 = Node(5)
    n2.left = n4
    n2.right = n5

    assert n1.levels == [[n1], [n2, n3], [n4, n5]]
    assert n1.leaves == [n3, n4, n5]
    assert n1.inorder == [n4, n2, n5, n1, n3]
    assert n1.preorder == [n1, n2, n4, n5, n3]
    assert n1.postorder == [n4, n5, n2, n3, n1]
    assert n1.levelorder == [n1, n2, n3, n4, n5]


def test_tree_list_representation():
    root = Node(1)
    assert root.values == [1]

    root.right = Node(3)
    assert root.values == [1, None, 3]

    root.left = Node(2)
    assert root.values == [1, 2, 3]

    root.right.left = Node(4)
    assert root.values == [1, 2, 3, None, None, 4]

    root.right.right = Node(5)
    assert root.values == [1, 2, 3, None, None, 4, 5]

    root.left.left = Node(6)
    assert root.values == [1, 2, 3, 6, None, 4, 5]

    root.left.right = Node(7)
    assert root.values == [1, 2, 3, 6, 7, 4, 5]
