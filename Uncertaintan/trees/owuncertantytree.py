"""Tree learner widget"""

import pandas as pd
import numpy as np
import scipy.sparse as sp

from collections import OrderedDict

from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.data.owcsvimport import pandas_to_table
from Orange.widgets.utils.localization import pl
from Orange.classification import _tree_scorers
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.classification import Learner
from Orange.base import TreeModel as TreeModelInterface
from Orange.tree import Node, DiscreteNode, MappedDiscreteNode, NumericNode
from Orange.statistics import distribution, contingency
from Orange.modelling import Fitter

class UncertanTreeModel(TreeModelInterface):
    """
    Tree classifier with proper handling of nominal attributes and binarization
    and the interface API for visualization.
    """

    def __init__(self, data, root):
        super().__init__(data.domain)
        self.instances = data
        self.root = root

        self._values = self._thresholds = self._code = None
        self._compile()
        self._compute_descriptions()

    def _prepare_predictions(self, n):
        rootval = self.root.value
        return np.empty((n,) + rootval.shape, dtype=rootval.dtype)

    def get_values_by_nodes(self, X):
        """Prediction that does not use compiled trees; for demo only"""
        n = len(X)
        y = self._prepare_predictions(n)
        for i in range(n):
            x = X[i]
            node = self.root
            while True:
                child_idx = node.descend(x)
                if np.isnan(child_idx):
                    break
                next_node = node.children[child_idx]
                if next_node is None:
                    break
                node = next_node
            y[i] = node.value
        return y

    def get_values_in_python(self, X):
        """Prediction with compiled code, but in Python; for demo only"""
        n = len(X)
        y = self._prepare_predictions(n)
        for i in range(n):
            x = X[i]
            node_ptr = 0
            while self._code[node_ptr]:
                val = x[self._code[node_ptr + 2]]
                if np.isnan(val):
                    break
                child_ptrs = self._code[node_ptr + 3:]
                if self._code[node_ptr] == 3:
                    node_idx = self._code[node_ptr + 1]
                    next_node_ptr = child_ptrs[int(val > self._thresholds[node_idx])]
                else:
                    next_node_ptr = child_ptrs[int(val)]
                if next_node_ptr == -1:
                    break
                node_ptr = next_node_ptr
            node_idx = self._code[node_ptr + 1]
            y[i] = self._values[node_idx]
        return y

    def get_values(self, X):
        from Orange.classification import _tree_scorers
        if sp.isspmatrix_csc(X):
            func = _tree_scorers.compute_predictions_csc
        elif sp.issparse(X):
            func = _tree_scorers.compute_predictions_csr
            X = X.tocsr()
        else:
            func = _tree_scorers.compute_predictions
        return func(X, self._code, self._values, self._thresholds)

    def predict(self, X):
        predictions = self.get_values(X)
        if self.domain.class_var.is_continuous:
            return predictions[:, 0]
        else:
            sums = np.sum(predictions, axis=1)
            # This can't happen because nodes with 0 instances are prohibited
            # zeros = (sums == 0)
            # predictions[zeros] = 1
            # sums[zeros] = predictions.shape[1]
            return predictions / sums[:, np.newaxis]

    def node_count(self):
        def _count(node):
            return 1 + sum(_count(c) for c in node.children if c)
        return _count(self.root)

    def depth(self):
        def _depth(node):
            return 1 + max((_depth(child) for child in node.children if child),
                           default=0)
        return _depth(self.root) - 1

    def leaf_count(self):
        def _count(node):
            return not node.children or \
                   sum(_count(c) if c else 1 for c in node.children)
        return _count(self.root)

    def get_instances(self, nodes):
        indices = self.get_indices(nodes)
        if indices is not None:
            return self.instances[indices]

    def get_indices(self, nodes):
        subsets = [node.subset for node in nodes]
        if subsets:
            return np.unique(np.hstack(subsets))

    @staticmethod
    def climb(node):
        while node:
            yield node
            node = node.parent

    @classmethod
    def rule(cls, node):
        rules = []
        used_attrs = set()
        for node in cls.climb(node):
            if node.parent is None or node.parent.attr_idx in used_attrs:
                continue
            parent = node.parent
            attr = parent.attr
            name = attr.name
            if isinstance(parent, NumericNode):
                lower, upper = node.condition
                if upper is None:
                    rules.append("{} > {}".format(name, attr.repr_val(lower)))
                elif lower is None:
                    rules.append("{} ≤ {}".format(name, attr.repr_val(upper)))
                else:
                    rules.append("{} < {} ≤ {}".format(
                        attr.repr_val(lower), name, attr.repr_val(upper)))
            else:
                rules.append("{}: {}".format(name, node.description))
            used_attrs.add(node.parent.attr_idx)
        return rules

    def print_tree(self, node=None, level=0):
        """String representation of tree for debug purposees"""
        if node is None:
            node = self.root
        res = ""
        for child in node.children:
            res += ("{:>20} {}{} {}\n".format(
                str(child.value), "    " * level, node.attr.name,
                child.description))
            res += self.print_tree(child, level + 1)
        return res

    NODE_TYPES = [Node, DiscreteNode, MappedDiscreteNode, NumericNode]

    def _compile(self):
        def _compute_sizes(node):
            nonlocal nnodes, codesize
            nnodes += 1
            codesize += 2  # node type + node index
            if isinstance(node, MappedDiscreteNode):
                codesize += len(node.mapping)
            if node.children:
                codesize += 1 + len(node.children)  # attr index + children ptrs
                for child in node.children:
                    if child is not None:
                        _compute_sizes(child)

        def _compile_node(node):
            from Orange.classification._tree_scorers import NULL_BRANCH

            # The node is compile into the following code (np.int32)
            # [0] node type: index of type in NODE_TYPES)
            # [1] node index: serves as index into values and thresholds
            # If the node is not a leaf:
            #     [2] attribute index
            # This is followed by an array of indices of the code for children
            # nodes. The length of this array is 2 for numeric attributes or
            # **the number of attribute values** for discrete attributes
            # This is different from the number of branches if discrete values
            # are mapped to branches

            # Thresholds and class distributions are stored in separate
            # 1-d and 2-d array arrays of type np.float, indexed by node index
            # The lengths of both equal the node count; we would gain (if
            # anything) by not reserving space for unused threshold space
            if node is None:
                return NULL_BRANCH
            nonlocal code_ptr, node_idx
            code_start = code_ptr
            self._code[code_ptr] = self.NODE_TYPES.index(type(node))
            self._code[code_ptr + 1] = node_idx
            code_ptr += 2

            self._values[node_idx] = node.value
            if isinstance(node, NumericNode):
                self._thresholds[node_idx] = node.threshold
            node_idx += 1

            # pylint: disable=unidiomatic-typecheck
            if type(node) == Node:
                return code_start

            self._code[code_ptr] = node.attr_idx
            code_ptr += 1

            jump_table_size = 2 if isinstance(node, NumericNode) \
                else len(node.attr.values)
            jump_table = self._code[code_ptr:code_ptr + jump_table_size]
            code_ptr += jump_table_size
            child_indices = [_compile_node(child) for child in node.children]
            if isinstance(node, MappedDiscreteNode):
                jump_table[:] = np.array(child_indices)[node.mapping]
            else:
                jump_table[:] = child_indices

            return code_start

        nnodes = codesize = 0
        _compute_sizes(self.root)
        print(nnodes)
        self._values = self._prepare_predictions(nnodes)
        self._thresholds = np.empty(nnodes)
        self._code = np.empty(codesize, np.int32)

        code_ptr = node_idx = 0
        _compile_node(self.root)

    def _compute_descriptions(self):
        def _compute_subtree(node):
            for i, child in enumerate(node.children):
                if child is None:
                    continue
                child.parent = node
                # These classes are friends
                # pylint: disable=protected-access
                node._set_child_descriptions(child, i, conditions)
                old_cond = conditions.get(node.attr)
                conditions[node.attr] = child.condition
                _compute_subtree(child)
                if old_cond is not None:
                    conditions[node.attr] = old_cond
                else:
                    del conditions[node.attr]

        conditions = OrderedDict()
        self.root.parent = None
        _compute_subtree(self.root)

    def predict_proba(self, data):
        return self(data, ret=TreeModelInterface.Probs)

    
class UncertanTree(Learner):
    """
    Tree inducer with proper handling of nominal attributes and binarization.

    The inducer can handle missing values of attributes and target.
    For discrete attributes with more than two possible values, each value can
    get a separate branch (`binarize=False`), or values can be grouped into
    two groups (`binarize=True`, default).

    The tree growth can be limited by the required number of instances for
    internal nodes and for leafs, the sufficient proportion of majority class,
    and by the maximal depth of the tree.

    If the tree is not binary, it can contain zero-branches.

    Args:
        binarize (bool):
            if `True` the inducer will find optimal split into two
            subsets for values of discrete attributes. If `False` (default),
            each value gets its branch.

        min_samples_leaf (float):
            the minimal number of data instances in a leaf

        min_samples_split (float):
            the minimal nubmer of data instances that is
            split into subgroups

        max_depth (int): the maximal depth of the tree

        sufficient_majority (float):
            a majority at which the data is not split
            further

    Returns:
        instance of OrangeTreeModel
    """
    __returns__ = UncertanTreeModel

    # Binarization is exhaustive, so we set a limit on the number of values
    MAX_BINARIZATION = 16

    def __init__(
            self, *args, binarize=False, max_depth=None,
            min_samples_leaf=1, min_samples_split=2, sufficient_majority=0.95,
            preprocessors=None, **kwargs):
        super().__init__(preprocessors=preprocessors)
        self.params = {}
        self.binarize = self.params['binarize'] = binarize
        self.min_samples_leaf = self.params['min_samples_leaf'] = min_samples_leaf
        self.min_samples_split = self.params['min_samples_split'] = min_samples_split
        self.sufficient_majority = self.params['sufficient_majority'] = sufficient_majority
        self.max_depth = self.params['max_depth'] = max_depth

    def _select_attr(self, data):
        """Select the attribute for the next split.

        Returns:
            tuple with an instance of Node and a numpy array indicating
            the branch index for each data instance, or -1 if data instance
            is dropped
        """
        # Prevent false warnings by pylint
        attr = attr_no = None
        col_x = None
        REJECT_ATTRIBUTE = 0, None, None, 0

        def _score_disc():
            """Scoring for discrete attributes, no binarization

            The class computes the entropy itself, not by calling other
            functions. This is to make sure that it uses the same
            definition as the below classes that compute entropy themselves
            for efficiency reasons."""
            n_values = len(attr.values)
            if n_values < 2:
                return REJECT_ATTRIBUTE

            cont = _tree_scorers.contingency(col_x, len(data.domain.attributes[attr_no].values),
                                             data.Y, len(data.domain.class_var.values))
            attr_distr = np.sum(cont, axis=0)
            null_nodes = attr_distr < self.min_samples_leaf
            # This is just for speed. If there is only a single non-null-node,
            # entropy wouldn't decrease anyway.
            if sum(null_nodes) >= n_values - 1:
                return REJECT_ATTRIBUTE
            cont[:, null_nodes] = 0
            attr_distr = np.sum(cont, axis=0)
            cls_distr = np.sum(cont, axis=1)
            n = np.sum(attr_distr)
            # Avoid log(0); <= instead of == because we need an array
            cls_distr[cls_distr <= 0] = 1
            attr_distr[attr_distr <= 0] = 1
            cont[cont <= 0] = 1
            class_entr = n * np.log(n) - np.sum(cls_distr * np.log(cls_distr))
            attr_entr = np.sum(attr_distr * np.log(attr_distr))
            cont_entr = np.sum(cont * np.log(cont))
            score = (class_entr - attr_entr + cont_entr) / n / np.log(2)
            score *= n / len(data)  # punishment for missing values
            branches = col_x.copy()
            branches[np.isnan(branches)] = -1
            if score == 0:
                return REJECT_ATTRIBUTE
            node = DiscreteNode(attr, attr_no, None)
            return score, node, branches, n_values

        def _score_disc_bin():
            """Scoring for discrete attributes, with binarization"""
            n_values = len(attr.values)
            if n_values <= 2:
                return _score_disc()
            cont = contingency.Discrete(data, attr)
            attr_distr = np.sum(cont, axis=0)
            # Skip instances with missing value of the attribute
            cls_distr = np.sum(cont, axis=1)
            if np.sum(attr_distr) == 0:  # all values are missing
                return REJECT_ATTRIBUTE
            best_score, best_mapping = _tree_scorers.find_binarization_entropy(
                cont, cls_distr, attr_distr, self.min_samples_leaf)
            if best_score <= 0:
                return REJECT_ATTRIBUTE
            best_score *= 1 - np.sum(cont.unknowns) / len(data)
            mapping, branches = MappedDiscreteNode.branches_from_mapping(
                col_x, best_mapping, n_values)
            node = MappedDiscreteNode(attr, attr_no, mapping, None)
            return best_score, node, branches, 2

        def _score_cont():
            """Scoring for numeric attributes"""
            nans = np.sum(np.isnan(col_x))
            non_nans = len(col_x) - nans
            arginds = np.argsort(col_x)[:non_nans]
            best_score, best_cut = _tree_scorers.find_threshold_entropy(
                col_x, data.Y, arginds,
                len(class_var.values), self.min_samples_leaf)
            if best_score == 0:
                return REJECT_ATTRIBUTE
            best_score *= non_nans / len(col_x)
            branches = np.full(len(col_x), -1, dtype=int)
            mask = ~np.isnan(col_x)
            branches[mask] = (col_x[mask] > best_cut).astype(int)
            node = NumericNode(attr, attr_no, best_cut, None)
            return best_score, node, branches, 2

        #######################################
        # The real _select_attr starts here
        is_sparse = sp.issparse(data.X)
        domain = data.domain
        class_var = domain.class_var
        best_score, *best_res = REJECT_ATTRIBUTE
        best_res = [Node(None, None, None)] + best_res[1:]
        disc_scorer = _score_disc_bin if self.binarize else _score_disc
        for attr_no, attr in enumerate(domain.attributes):
            col_x = data.X[:, attr_no]
            if is_sparse:
                col_x = col_x.toarray()
                col_x = col_x.flatten()
            sc, *res = disc_scorer() if attr.is_discrete else _score_cont()
            if res[0] is not None and sc > best_score:
                best_score, best_res = sc, res
        best_res[0].value = distribution.Discrete(data, class_var)
        return best_res

    def _build_tree(self, data, active_inst, level=1):
        """Induce a tree from the given data

        Returns:
            root node (Node)"""
        node_insts = data[active_inst]
        distr = distribution.Discrete(node_insts, data.domain.class_var)
        if len(node_insts) < self.min_samples_leaf:
            return None
        if len(node_insts) < self.min_samples_split or \
                max(distr) >= sum(distr) * self.sufficient_majority or \
                self.max_depth is not None and level > self.max_depth:
            node, branches, n_children = Node(None, None, distr), None, 0
        else:
            node, branches, n_children = self._select_attr(node_insts)
        node.subset = active_inst
        if branches is not None:
            node.children = [
                self._build_tree(data, active_inst[branches == br], level + 1)
                for br in range(n_children)]
        return node

    def fit_storage(self, data):
        if self.binarize and any(
                attr.is_discrete and len(attr.values) > self.MAX_BINARIZATION
                for attr in data.domain.attributes):
            # No fallback in the script; widgets can prevent this error
            # by providing a fallback and issue a warning about doing so
            raise ValueError("Exhaustive binarization does not handle "
                             "attributes with more than {} values".
                             format(self.MAX_BINARIZATION))

        active_inst = np.nonzero(~np.isnan(data.Y))[0].astype(np.int32)
        root = self._build_tree(data, active_inst)
        if root is None:
            distr = distribution.Discrete(data, data.domain.class_var)
            if np.sum(distr) == 0:
                distr[:] = 1
            root = Node(None, 0, distr)
        root.subset = active_inst
        model = UncertanTreeModel(data, root)
        return model

class OWUncertaintyTree(OWBaseLearner):
    """Tree algorithm with forward pruning."""
    name = "Uncertainty Tree"
    description = "A tree algorithm for data with uncertainties"
    icon = "icons/Tree.svg"
    priority = 30
    keywords = "tree, uncertainty tree"

    LEARNER = UncertanTree

    binary_trees = Setting(True)
    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)

    # Classification only settings
    limit_majority = Setting(True)
    sufficient_majority = Setting(95)

    spin_boxes = (
        ("Min. number of instances in leaves: ",
         "limit_min_leaf", "min_leaf", 1, 1000),
        ("Do not split subsets smaller than: ",
         "limit_min_internal", "min_internal", 1, 1000),
        ("Limit the maximal tree depth to: ",
         "limit_depth", "max_depth", 1, 1000))

    classification_spin_boxes = (
        ("Stop when majority reaches [%]: ",
         "limit_majority", "sufficient_majority", 51, 100),)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, 'Parameters')
        # the checkbox is put into vBox for alignemnt with other checkboxes
        gui.checkBox(box, self, "binary_trees", "Induce binary tree",
                     callback=self.settings_changed,
                     attribute=Qt.WA_LayoutUsesWidgetRect)
        for label, check, setting, fromv, tov in self.spin_boxes:
            gui.spin(box, self, setting, fromv, tov, label=label,
                     checked=check, alignment=Qt.AlignRight,
                     callback=self.settings_changed,
                     checkCallback=self.settings_changed, controlWidth=80)

    def add_classification_layout(self, box):
        for label, check, setting, minv, maxv in self.classification_spin_boxes:
            gui.spin(box, self, setting, minv, maxv,
                     label=label, checked=check, alignment=Qt.AlignRight,
                     callback=self.settings_changed, controlWidth=80,
                     checkCallback=self.settings_changed)

    def learner_kwargs(self):
        # Pylint doesn't get our Settings
        # pylint: disable=invalid-sequence-index
        return dict(
            max_depth=(None, self.max_depth)[self.limit_depth],
            min_samples_split=(2, self.min_internal)[self.limit_min_internal],
            min_samples_leaf=(1, self.min_leaf)[self.limit_min_leaf],
            binarize=self.binary_trees,
            preprocessors=self.preprocessors,
            sufficient_majority=(1, self.sufficient_majority / 100)[
                self.limit_majority])

    def create_learner(self):
        print("create_learner()")
        # pylint: disable=not-callable
        return self.LEARNER(**self.learner_kwargs())

    def get_learner_parameters(self):
        # from Orange.widgets.report import plural_w
        print("get_learner_parameters()")
        items = OrderedDict()
        items["Pruning"] = ", ".join(s for s, c in (
            (f'at least {self.min_leaf} '
             f'{pl(self.min_leaf, "instance")} in leaves',
             self.limit_min_leaf),
            (f'at least {self.min_internal} '
             f'{pl(self.min_internal, "instance")} in internal nodes',
             self.limit_min_internal),
            (f'maximum depth {self.max_depth}',
             self.limit_depth)
        ) if c) or "None"
        if self.limit_majority:
            items["Splitting"] = "Stop splitting when majority reaches %d%% " \
                                 "(classification only)" % \
                                 self.sufficient_majority
        items["Binary trees"] = ("No", "Yes")[self.binary_trees]
        return items


if __name__ == "__main__":  # pragma: no cover
    df = pd.read_csv("data.csv")
    print(df)
    WidgetPreview(OWUncertaintyTree).run(Table("iris"))