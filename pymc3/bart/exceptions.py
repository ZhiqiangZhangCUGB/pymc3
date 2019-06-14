class BinaryTreeError(Exception):
    """Base (catch-all) binarytree exception."""


class NodeIndexError(BinaryTreeError):
    """Node index was invalid."""


class NodeValueError(BinaryTreeError):
    """Node value was not a number (e.g. int, float)."""


class NodeSplitVariableIndexError(BinaryTreeError):
    """Split node splitting variable index was invalid."""


class NodeSplitVariableTypeError(BinaryTreeError):
    """Split node splitting variable type was invalid."""


class NodeQuantitativeSplitValueError(BinaryTreeError):
    """Split node value was not a number (e.g. int, float)."""


class NodeQualitativeSplitValueError(BinaryTreeError):
    """Split node value was not a set."""


class LeafNodeValueError(BinaryTreeError):
    """Leaf node value was not float."""
