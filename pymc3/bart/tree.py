import numbers
import math
from pymc3.bart.exceptions import (
    NodeIndexError,
    NodeSplitVariableIndexError,
    NodeSplitVariableTypeError,
    NodeQuantitativeSplitValueError,
    NodeQualitativeSplitValueError,
    LeafNodeValueError,
)

class Tree:
    '''
    Full binary tree
    '''
    def __init__(self):
        # permite eliminar nodos que son hojas sin tener que corregir el indice de todos los nodos con idx mas alto.
        self.tree_structure = {}
        self.num_nodes = 0
        self.tree_depth = -1
        self.idx_leaves_nodes = []


class BaseNode:
    def __init__(self, index):
        if not isinstance(index, int) or index < 0:
            raise NodeIndexError('node index must be a non-negative int')
        self.index = index
        self.depth = int(math.floor(math.log(index+1, 2)))


class SplitNode(BaseNode):
    def __init__(self, index, idx_split_variable, type_split_variable, split_value):
        super().__init__(index)

        if not isinstance(idx_split_variable, int) or idx_split_variable < 0:
            raise NodeSplitVariableIndexError('index of split variable must be a non-negative int')
        if type_split_variable is not 'quantitative' and type_split_variable is not 'qualitative':
            raise NodeSplitVariableTypeError('type of split variable must be "quantitative" or "qualitative"')
        if type_split_variable is 'quantitative':
            if not isinstance(split_value, numbers.Number):
                raise NodeQuantitativeSplitValueError('node split value must be a number')
        else:
            if not isinstance(split_value, set):
                raise NodeQualitativeSplitValueError('node split value must be a set')

        self.idx_split_variable = idx_split_variable
        self.type_split_variable = type_split_variable
        self.split_value = split_value
        self.operator = '<=' if self.type_split_variable == 'quantitative' else 'in'

    def __repr__(self):
        return 'SplitNode(index={}, idx_split_variable={}, type_split_variable={!r}, ' \
               'split_value={})'.format(self.index, self.idx_split_variable,
                                        self.type_split_variable, self.split_value)

    def __str__(self):
        return 'x[{}] {} {}'.format(self.idx_split_variable, self.operator, self.split_value)


class LeafNode(BaseNode):
    def __init__(self, index, value):
        super().__init__(index)
        if not isinstance(value, float):
            raise LeafNodeValueError('leaf node value must be float')
        self.value = value

    def __repr__(self):
        return 'LeafNode(index={}, value={})'.format(self.index, self.value)

    def __str__(self):
        return '{}'.format(self.value)
