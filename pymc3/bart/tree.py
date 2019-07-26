import numpy as np
import math
from pymc3.bart.exceptions import (
    TreeStructureError,
    TreeNodeError,
)


class Tree:
    ''' Full binary tree'''
    def __init__(self):
        self.tree_structure = {}
        self.num_nodes = 0
        self.idx_leaf_nodes = []
        self.idx_prunable_split_nodes = []

    def __getitem__(self, index):
        return self.get_node(index)

    def __setitem__(self, index, node):
        self.set_node(index, node)

    def __delitem__(self, index):
        self.delete_node(index)

    def __iter__(self):
        return iter(self.tree_structure)

    def __eq__(self, other):
        return self.tree_structure == other.tree_structure and self.num_nodes == other.num_nodes\
               and set(self.idx_leaf_nodes) == set(other.idx_leaf_nodes)\
               and set(self.idx_prunable_split_nodes) == set(other.idx_prunable_split_nodes)

    def __hash__(self):
        return 0

    def __len__(self):
        return len(self.tree_structure)

    def __repr__(self):
        return 'Tree(num_nodes={})'.format(self.num_nodes)

    def __str__(self):
        lines = self._build_tree_string(index=0, show_index=False, delimiter='-')[0]
        return '\n' + '\n'.join((line.rstrip() for line in lines))

    def _build_tree_string(self, index, show_index=False, delimiter='-'):
        """Recursively walk down the binary tree and build a pretty-print string.

        In each recursive call, a "box" of characters visually representing the
        current (sub)tree is constructed line by line. Each line is padded with
        whitespaces to ensure all lines in the box have the same length. Then the
        box, its width, and start-end positions of its root node value repr string
        (required for drawing branches) are sent up to the parent call. The parent
        call then combines its left and right sub-boxes to build a larger box etc.
        """
        if index not in self.tree_structure:
            return [], 0, 0, 0

        line1 = []
        line2 = []
        current_node = self.get_node(index)
        if show_index:
            node_repr = '{}{}{}'.format(index, delimiter, str(current_node))
        else:
            node_repr = str(current_node)

        new_root_width = gap_size = len(node_repr)

        left_child = current_node.get_idx_left_child()
        right_child = current_node.get_idx_right_child()

        # Get the left and right sub-boxes, their widths, and root repr positions
        l_box, l_box_width, l_root_start, l_root_end = \
            self._build_tree_string(left_child, show_index, delimiter)
        r_box, r_box_width, r_root_start, r_root_end = \
            self._build_tree_string(right_child, show_index, delimiter)

        # Draw the branch connecting the current root node to the left sub-box
        # Pad the line with whitespaces where necessary
        if l_box_width > 0:
            l_root = (l_root_start + l_root_end) // 2 + 1
            line1.append(' ' * (l_root + 1))
            line1.append('_' * (l_box_width - l_root))
            line2.append(' ' * l_root + '/')
            line2.append(' ' * (l_box_width - l_root))
            new_root_start = l_box_width + 1
            gap_size += 1
        else:
            new_root_start = 0

        # Draw the representation of the current root node
        line1.append(node_repr)
        line2.append(' ' * new_root_width)

        # Draw the branch connecting the current root node to the right sub-box
        # Pad the line with whitespaces where necessary
        if r_box_width > 0:
            r_root = (r_root_start + r_root_end) // 2
            line1.append('_' * r_root)
            line1.append(' ' * (r_box_width - r_root + 1))
            line2.append(' ' * r_root + '\\')
            line2.append(' ' * (r_box_width - r_root))
            gap_size += 1
        new_root_end = new_root_start + new_root_width - 1

        # Combine the left and right sub-boxes with the branches drawn above
        gap = ' ' * gap_size
        new_box = [''.join(line1), ''.join(line2)]
        for i in range(max(len(l_box), len(r_box))):
            l_line = l_box[i] if i < len(l_box) else ' ' * l_box_width
            r_line = r_box[i] if i < len(r_box) else ' ' * r_box_width
            new_box.append(l_line + gap + r_line)

        # Return the new box, its width and its root repr positions
        return new_box, len(new_box[0]), new_root_start, new_root_end

    def get_node(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError('Node index must be a non-negative int')
        if index not in self.tree_structure:
            raise TreeStructureError('Node missing at index {}'.format(index))
        return self.tree_structure[index]

    def set_node(self, index, node):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError('Node index must be a non-negative int')
        if not isinstance(node, SplitNode) and not isinstance(node, LeafNode):
            raise TreeStructureError('Node class must be SplitNode or LeafNode')
        if index in self.tree_structure:
            raise TreeStructureError('Node index already exist in tree')
        if self.num_nodes == 0 and index != 0:
            raise TreeStructureError('Root node must have index zero')
        parent_index = node.get_idx_parent_node()
        if self.num_nodes != 0 and parent_index not in self.tree_structure:
            raise TreeStructureError('Invalid index, node must have a parent node')
        if self.num_nodes != 0 and not isinstance(self.get_node(parent_index), SplitNode):
            raise TreeStructureError('Parent node must be of class SplitNode')
        if index != node.index:
            raise TreeStructureError('Node must have same index as tree index')
        self.tree_structure[index] = node
        self.num_nodes += 1
        if isinstance(node, LeafNode):
            self.idx_leaf_nodes.append(index)

    def delete_node(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError('Node index must be a non-negative int')
        if index not in self.tree_structure:
            raise TreeStructureError('Node missing at index {}'.format(index))
        current_node = self.get_node(index)
        left_child_idx = current_node.get_idx_left_child()
        right_child_idx = current_node.get_idx_right_child()
        if left_child_idx in self.tree_structure or right_child_idx in self.tree_structure:
            raise TreeStructureError('Invalid removal of node, leaving two orphans nodes')
        if isinstance(current_node, LeafNode):
            self.idx_leaf_nodes.remove(index)
        del self.tree_structure[index]
        self.num_nodes -= 1

    def make_digraph(self, name=''):
        """Make graphviz Digraph of Tree

        Returns
        -------
        graphviz.Digraph
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError('This function requires the python library graphviz, along with binaries. '
                              'The easiest way to install all of this is by running\n\n'
                              '\tconda install -c conda-forge python-graphviz')
        graph = graphviz.Digraph(name)
        graph = self._digraph_tree_traversal(0, graph)
        return graph

    def _digraph_tree_traversal(self, index, graph):
        if index not in self.tree_structure.keys():
            return graph
        current_node = self.get_node(index)
        style = ''
        if isinstance(current_node, SplitNode):
            shape = 'box'
            if index in self.idx_prunable_split_nodes:
                style = 'filled'
        else:
            shape = 'ellipse'
        graph.node(name=str(index), label=str(current_node), shape=shape, style=style)

        parent_index = current_node.get_idx_parent_node()
        if parent_index in self.tree_structure:
            if current_node.is_left_child():
                graph.edge(tail_name=str(parent_index), head_name=str(index), label='T')
            else:
                graph.edge(tail_name=str(parent_index), head_name=str(index), label='F')

        left_child = current_node.get_idx_left_child()
        right_child = current_node.get_idx_right_child()
        graph = self._digraph_tree_traversal(left_child, graph)
        graph = self._digraph_tree_traversal(right_child, graph)

        return graph

    def out_of_sample_predict(self, x):
        leaf_node = self._traverse_tree(x=x, node_index=0)
        return leaf_node.value

    def predict_output(self, num_observations):
        output = np.zeros(num_observations)
        for node_index in self.idx_leaf_nodes:
            current_node = self.get_node(node_index)
            output[current_node.idx_data_points] = current_node.value
        return output

    def _traverse_tree(self, x, node_index=0):
        current_node = self.get_node(node_index)
        if isinstance(current_node, SplitNode):
            if current_node.evaluate_splitting_rule(x):
                left_child = current_node.get_idx_left_child()
                final_node = self._traverse_tree(x, left_child)
            else:
                right_child = current_node.get_idx_right_child()
                final_node = self._traverse_tree(x, right_child)
        else:
            final_node = current_node
        return final_node

    def prior_log_probability_tree(self, alpha, beta):
        prior_log_probability = 0.0
        for idx, node in self.tree_structure.items():
            prior_log_probability += node.prior_log_probability_node(alpha, beta)
        return prior_log_probability

    def grow_tree(self, index_leaf_node, new_split_node, new_left_node, new_right_node):
        current_node = self.get_node(index_leaf_node)
        if not isinstance(current_node, LeafNode):
            raise TreeStructureError('The tree grows from the leaves')
        if not isinstance(new_split_node, SplitNode):
            raise TreeStructureError('The node that replaces the leaf node must be SplitNode')
        if not isinstance(new_left_node, LeafNode) or not isinstance(new_right_node, LeafNode):
            raise TreeStructureError('The new leaves must be LeafNode')

        self.delete_node(index_leaf_node)
        self.set_node(index_leaf_node, new_split_node)
        self.set_node(new_left_node.index, new_left_node)
        self.set_node(new_right_node.index, new_right_node)

        parent_index = current_node.get_idx_parent_node()
        self.idx_prunable_split_nodes.append(index_leaf_node)
        if parent_index in self.idx_prunable_split_nodes:
            self.idx_prunable_split_nodes.remove(parent_index)

    def prune_tree(self, index_split_node, new_leaf_node):
        current_node = self.get_node(index_split_node)
        if not isinstance(current_node, SplitNode):
            raise TreeStructureError('Only SplitNodes are prunable')
        if not self.is_node_prunable(index_split_node):
            raise TreeStructureError('SplitNodes must have two LeafNodes as children to be prunable')

        left_child_idx = current_node.get_idx_left_child()
        right_child_idx = current_node.get_idx_right_child()
        parent_index = current_node.get_idx_parent_node()

        self.delete_node(left_child_idx)
        self.delete_node(right_child_idx)
        self.delete_node(index_split_node)

        self.set_node(index_split_node, new_leaf_node)

        self.idx_prunable_split_nodes.remove(index_split_node)
        if self.is_node_prunable(parent_index):
            self.idx_prunable_split_nodes.append(parent_index)

    def is_node_prunable(self, idx):
        '''
        A splitting node is prunable if both its children are leaf nodes.
        '''
        current_node = self.get_node(idx)
        left_child = self.get_node(current_node.get_idx_left_child())
        right_child = self.get_node(current_node.get_idx_right_child())

        return True if isinstance(left_child, LeafNode) and isinstance(right_child, LeafNode) else False

    @staticmethod
    def init_tree(leaf_node_value, idx_data_points):
        new_tree = Tree()
        new_tree[0] = LeafNode(index=0, value=leaf_node_value, idx_data_points=idx_data_points)
        return new_tree

    def get_tree_depth(self):
        max_depth = 0
        for idx_leaf_node in self.idx_leaf_nodes:
            leaf_node = self.get_node(idx_leaf_node)
            if max_depth < leaf_node.depth:
                max_depth = leaf_node.depth
        return max_depth


class BaseNode:
    def __init__(self, index, idx_data_points):
        if not isinstance(index, int) or index < 0:
            raise TreeNodeError('Node index must be a non-negative int')
        if not isinstance(idx_data_points, np.ndarray) or idx_data_points.dtype.type is not np.int64:
            raise TreeNodeError('Index of data points must be a numpy.ndarray of integers')
        if len(idx_data_points) == 0:
            raise TreeNodeError('Index of data points can not be empty')
        self.index = index
        self.depth = int(math.floor(math.log(index+1, 2)))
        self.idx_data_points = idx_data_points

    def __eq__(self, other):
        return self.index == other.index and self.depth == other.depth and \
               np.array_equal(self.idx_data_points, other.idx_data_points)

    def get_idx_parent_node(self):
        return (self.index - 1) // 2

    def get_idx_left_child(self):
        return self.index * 2 + 1

    def get_idx_right_child(self):
        return self.get_idx_left_child() + 1

    def is_left_child(self):
        return bool(self.index % 2)

    def get_idx_sibling(self):
        return (self.index + 1) if self.is_left_child() else (self.index - 1)


class SplitNode(BaseNode):
    def __init__(self, index, idx_split_variable, type_split_variable, split_value, idx_data_points):
        super().__init__(index, idx_data_points)

        if not isinstance(idx_split_variable, int) or idx_split_variable < 0:
            raise TreeNodeError('Index of split variable must be a non-negative int')
        if type_split_variable is not 'quantitative' and type_split_variable is not 'qualitative':
            raise TreeNodeError('Type of split variable must be "quantitative" or "qualitative"')
        if type_split_variable is 'quantitative':
            if not isinstance(split_value, float):
                raise TreeNodeError('Node split value type must be float')
        else:
            if not isinstance(split_value, set):
                raise TreeNodeError('Node split value must be a set')

        self.idx_split_variable = idx_split_variable
        self.type_split_variable = type_split_variable
        self.split_value = split_value
        self.operator = '<=' if self.type_split_variable == 'quantitative' else 'in'

    def __repr__(self):
        return 'SplitNode(index={}, idx_split_variable={}, type_split_variable={!r}, ' \
               'split_value={}, len(idx_data_points)={})'\
            .format(self.index, self.idx_split_variable, self.type_split_variable,
                    self.split_value, len(self.idx_data_points))

    def __str__(self):
        return 'x[{}] {} {}'.format(self.idx_split_variable, self.operator, self.split_value)

    def __eq__(self, other):
        if isinstance(other, SplitNode):
            return super().__eq__(other) and self.idx_split_variable == other.idx_split_variable \
                   and self.type_split_variable == other.type_split_variable \
                   and self.split_value == other.split_value \
                   and self.operator == other.operator
        else:
            return NotImplemented

    def evaluate_splitting_rule(self, x):
        if x is np.NaN:
            return False
        elif self.type_split_variable == 'quantitative':
            return x[self.idx_split_variable] <= self.split_value
        else:
            return x[self.idx_split_variable] in self.split_value

    def prior_log_probability_node(self, alpha, beta):
        return np.log(alpha * np.power(1.0 + self.depth, -beta))


class LeafNode(BaseNode):
    def __init__(self, index, value, idx_data_points):
        super().__init__(index, idx_data_points)
        if not isinstance(value, float):
            raise TreeNodeError('Leaf node value type must be float')
        self.value = value

    def __repr__(self):
        return 'LeafNode(index={}, value={}, len(idx_data_points)={})'.format(self.index, self.value,
                                                                              len(self.idx_data_points))

    def __str__(self):
        return '{}'.format(self.value)

    def __eq__(self, other):
        if isinstance(other, LeafNode):
            return super().__eq__(other) and self.value == other.value
        else:
            return NotImplemented

    def prior_log_probability_node(self, alpha, beta):
        return np.log(1.0 - alpha * np.power(1.0 + self.depth, -beta))
