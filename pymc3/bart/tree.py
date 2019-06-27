import numbers
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

    def __getitem__(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError('Node index must be a non-negative int')
        if index not in self.tree_structure:
            raise TreeStructureError('Node missing at index {}'.format(index))
        return self.tree_structure[index]

    def __setitem__(self, index, node):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError('Node index must be a non-negative int')
        if not isinstance(node, SplitNode) and not isinstance(node, LeafNode):
            raise TreeStructureError('Node class must be SplitNode or LeafNode')
        if index in self.tree_structure:
            raise TreeStructureError('Node index already exist in tree')
        if self.num_nodes == 0 and index != 0:
            raise TreeStructureError('Root node must have index zero')
        parent_index = (index - 1) // 2
        if self.num_nodes != 0 and parent_index not in self.tree_structure:
            raise TreeStructureError('Node must have a parent node')
        if self.num_nodes != 0 and not isinstance(self.tree_structure[parent_index], SplitNode):
            raise TreeStructureError('Parent node must be of class SplitNode')
        if index != node.index:
            raise TreeStructureError('Node must have same index as tree index')
        self.tree_structure[index] = node
        self.num_nodes += 1
        if isinstance(node, LeafNode):
            self.idx_leaf_nodes.append(index)

    def __delitem__(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeStructureError('Node index must be a non-negative int')
        if index not in self.tree_structure:
            raise TreeStructureError('Node missing at index {}'.format(index))
        possible_child = index * 2 + 1
        if possible_child in self.tree_structure:
            raise TreeStructureError('Invalid removal of node, leaving orphan '
                                     'children at index {}'.format(possible_child))
        del self.tree_structure[index]
        self.num_nodes -= 1
        if index in self.idx_leaf_nodes:
            self.idx_leaf_nodes.remove(index)

    def __iter__(self):
        return iter(self.tree_structure)

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
        if show_index:
            node_repr = '{}{}{}'.format(index, delimiter, str(self.tree_structure[index]))
        else:
            node_repr = str(self.tree_structure[index])

        new_root_width = gap_size = len(node_repr)

        left_child = index * 2 + 1
        right_child = index * 2 + 2

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
        graph = self._tree_traversal(0, graph)
        return graph

    def _tree_traversal(self, index, graph):
        if index not in self.tree_structure.keys():
            return graph
        node = self.tree_structure[index]
        if isinstance(node, SplitNode):
            shape = 'box'
        else:
            shape = 'ellipse'
        graph.node(name=str(index), label=str(node), shape=shape)

        parent_index = (index - 1) // 2
        is_left_child = index % 2
        if parent_index in self.tree_structure:
            if is_left_child:
                graph.edge(tail_name=str(parent_index), head_name=str(index), label='T')
            else:
                graph.edge(tail_name=str(parent_index), head_name=str(index), label='F')

        left_child = index * 2 + 1
        right_child = index * 2 + 2
        graph = self._tree_traversal(left_child, graph)
        graph = self._tree_traversal(right_child, graph)

        return graph

    def traverse_tree(self, x, node_index=0):
        current_node = self.tree_structure[node_index]
        if isinstance(current_node, SplitNode):
            if current_node.evaluate_splitting_rule(x):
                left_child = node_index * 2 + 1
                final_node = self.traverse_tree(x, left_child)
            else:
                right_child = node_index * 2 + 2
                final_node = self.traverse_tree(x, right_child)
        else:
            final_node = current_node
        return final_node

    def out_of_sample_predict(self, x):
        leaf_node = self.traverse_tree(x=x, node_index=0)
        return leaf_node.value


class BaseNode:
    def __init__(self, index):
        if not isinstance(index, int) or index < 0:
            raise TreeNodeError('Node index must be a non-negative int')
        self.index = index
        self.depth = int(math.floor(math.log(index+1, 2)))

    def __eq__(self, other):
        return self.index == other.index and self.depth == other.depth


class SplitNode(BaseNode):
    def __init__(self, index, idx_split_variable, type_split_variable, split_value):
        super().__init__(index)

        if not isinstance(idx_split_variable, int) or idx_split_variable < 0:
            raise TreeNodeError('Index of split variable must be a non-negative int')
        if type_split_variable is not 'quantitative' and type_split_variable is not 'qualitative':
            raise TreeNodeError('Type of split variable must be "quantitative" or "qualitative"')
        if type_split_variable is 'quantitative':
            if not isinstance(split_value, numbers.Number):
                raise TreeNodeError('Node split value must be a number')
        else:
            if not isinstance(split_value, set):
                raise TreeNodeError('Node split value must be a set')

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

    def __eq__(self, other):
        return super().__eq__(other) and self.idx_split_variable == other.idx_split_variable \
               and self.type_split_variable == other.type_split_variable \
               and self.split_value == other.split_value \
               and self.operator == other.operator

    def evaluate_splitting_rule(self, x):
        if self.type_split_variable == 'quantitative':
            return x[self.idx_split_variable] <= self.split_value
        else:
            return x[self.idx_split_variable] in self.split_value


class LeafNode(BaseNode):
    def __init__(self, index, value):
        super().__init__(index)
        if not isinstance(value, float):
            raise TreeNodeError('Leaf node value must be float')
        self.value = value

    def __repr__(self):
        return 'LeafNode(index={}, value={})'.format(self.index, self.value)

    def __str__(self):
        return '{}'.format(self.value)

    def __eq__(self, other):
        return super().__eq__(other) and self.value == other.value
