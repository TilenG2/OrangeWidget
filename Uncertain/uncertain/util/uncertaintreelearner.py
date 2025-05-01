import numpy as np
import scipy.sparse as sp

from Orange.classification import _tree_scorers
from Orange.classification import Learner
from Orange.tree import Node, NumericNode, TreeModel
from Orange.statistics import distribution
    
class UncertainTreeLearner(Learner):
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
    __returns__ = TreeModel

    # Binarization is exhaustive, so we set a limit on the number of values
    MAX_BINARIZATION = 16

    def __init__(
            self, *args, max_depth=None,
            min_samples_leaf=1, min_samples_split=2, sufficient_majority=0.95,
            preprocessors=None, uncertainty_multiplyer=0.5, post_hoc=True, **kwargs):
        super().__init__(preprocessors=preprocessors)
        self.params = {}
        self.min_samples_leaf = self.params['min_samples_leaf'] = min_samples_leaf
        self.min_samples_split = self.params['min_samples_split'] = min_samples_split
        self.sufficient_majority = self.params['sufficient_majority'] = sufficient_majority
        self.max_depth = self.params['max_depth'] = max_depth
        self.uncertainty_multiplyer = uncertainty_multiplyer
        self.post_hoc = post_hoc

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

        # def _score_disc():
        #     """Scoring for discrete attributes, no binarization

        #     The class computes the entropy itself, not by calling other
        #     functions. This is to make sure that it uses the same
        #     definition as the below classes that compute entropy themselves
        #     for efficiency reasons."""
        #     n_values = len(attr.values)
        #     if n_values < 2:
        #         return REJECT_ATTRIBUTE

        #     cont = _tree_scorers.contingency(col_x, len(data.domain.attributes[attr_no].values),
        #                                      data.Y, len(data.domain.class_var.values))
        #     attr_distr = np.sum(cont, axis=0)
        #     null_nodes = attr_distr < self.min_samples_leaf
        #     # This is just for speed. If there is only a single non-null-node,
        #     # entropy wouldn't decrease anyway.
        #     if sum(null_nodes) >= n_values - 1:
        #         return REJECT_ATTRIBUTE
        #     cont[:, null_nodes] = 0
        #     attr_distr = np.sum(cont, axis=0)
        #     cls_distr = np.sum(cont, axis=1)
        #     n = np.sum(attr_distr)
        #     # Avoid log(0); <= instead of == because we need an array
        #     cls_distr[cls_distr <= 0] = 1
        #     attr_distr[attr_distr <= 0] = 1
        #     cont[cont <= 0] = 1
        #     class_entr = n * np.log(n) - np.sum(cls_distr * np.log(cls_distr))
        #     attr_entr = np.sum(attr_distr * np.log(attr_distr))
        #     cont_entr = np.sum(cont * np.log(cont))
        #     score = (class_entr - attr_entr + cont_entr) / n / np.log(2)
        #     score *= n / len(data)  # punishment for missing values
        #     branches = col_x.copy()
        #     branches[np.isnan(branches)] = -1
        #     if score == 0:
        #         return REJECT_ATTRIBUTE
        #     node = DiscreteNode(attr, attr_no, None)
        #     return score, node, branches, n_values

        # def _score_disc_bin():
        #     """Scoring for discrete attributes, with binarization"""
        #     n_values = len(attr.values)
        #     if n_values <= 2:
        #         return _score_disc()
        #     cont = contingency.Discrete(data, attr)
        #     attr_distr = np.sum(cont, axis=0)
        #     # Skip instances with missing value of the attribute
        #     cls_distr = np.sum(cont, axis=1)
        #     if np.sum(attr_distr) == 0:  # all values are missing
        #         return REJECT_ATTRIBUTE
        #     best_score, best_mapping = _tree_scorers.find_binarization_entropy(
        #         cont, cls_distr, attr_distr, self.min_samples_leaf)
        #     if best_score <= 0:
        #         return REJECT_ATTRIBUTE
        #     best_score *= 1 - np.sum(cont.unknowns) / len(data)
        #     mapping, branches = MappedDiscreteNode.branches_from_mapping(
        #         col_x, best_mapping, n_values)
        #     node = MappedDiscreteNode(attr, attr_no, mapping, None)
        #     return best_score, node, branches, 2

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
            
            # feature_with_metas = []
            # for X, meta in zip(data.X, data.metas):
            #     feature_with_metas.append([X[attr_no], meta[attr_no]])
            # feature_with_metas = np.array(feature_with_metas)
            
            #TODO generalize this
            feature_with_metas = np.hstack((
                data.get_column(attr_no)[:, None],
                data.get_column(domain.attributes[attr_no].name.replace("Observed Value", "Uncertainty"))[:, None]
            ))
            
            feature_with_metas = feature_with_metas[feature_with_metas[:, 0].argsort()]
            index_best_cut = np.where(feature_with_metas[:, 0] == best_cut)[0][-1]
            # Ajutst the cut based on the uncertainty
            if self.post_hoc:
                best_cut = (feature_with_metas[index_best_cut][0] + feature_with_metas[index_best_cut+1][0]) * 0.5

                best_score *= non_nans / len(col_x)
                branches = np.full(len(col_x), -1, dtype=int)
                mask = ~np.isnan(col_x)
                branches[mask] = (col_x[mask] > best_cut).astype(int)

                offset = (-feature_with_metas[index_best_cut][1] + feature_with_metas[index_best_cut+1][1]) * self.uncertainty_multiplyer
                best_cut = best_cut + offset
            else:
                # used_indexes = []
                # shift = False
                best_cut = (feature_with_metas[index_best_cut][0] + feature_with_metas[index_best_cut+1][0]) * 0.5
                offset = (-feature_with_metas[index_best_cut][1] + feature_with_metas[index_best_cut+1][1]) * self.uncertainty_multiplyer
                best_cut = best_cut + offset
                    
                best_score *= non_nans / len(col_x)
                branches = np.full(len(col_x), -1, dtype=int)
                mask = ~np.isnan(col_x)
                branches[mask] = (col_x[mask] > best_cut).astype(int)
                    # AAAsuma = np.sum(branches)
                    # AAlenght = len(branches)
                    # Aresult = len(branches) - np.sum(branches)
                    # if len(branches) - np.sum(branches) == 0 or np.sum(branches) == 0:
                    #     shift = True
                    # if np.sum(branches) < self.min_samples_leaf: # premal elementu gre v desnga
                    #     index_best_cut += -1 if shift else +1
                    #     if index_best_cut < 0 or index_best_cut + 1 >= len(branches):
                    #         return REJECT_ATTRIBUTE
                    #     if index_best_cut in used_indexes:
                    #         return REJECT_ATTRIBUTE
                    #     used_indexes.append(index_best_cut)
                    #     shift = True
                    #     continue
                    # elif len(branches) - np.sum(branches) < self.min_samples_leaf : # premaw elementov gre v levga
                    #     index_best_cut += +1 if shift else -1
                    #     if index_best_cut < 0 or index_best_cut + 1 >= len(branches):
                    #         return REJECT_ATTRIBUTE
                    #     if index_best_cut in used_indexes:
                    #         return REJECT_ATTRIBUTE
                    #     used_indexes.append(index_best_cut)
                    #     shift = True
                    #     continue
                    # else: break
            node = NumericNode(attr, attr_no, best_cut, None)
            return best_score, node, branches, 2

        #######################################
        # The real _select_attr starts here
        is_sparse = sp.issparse(data.X)
        domain = data.domain
        class_var = domain.class_var
        best_score, *best_res = REJECT_ATTRIBUTE
        best_res = [Node(None, None, None)] + best_res[1:]
        # disc_scorer = _score_disc_bin if self.binarize else _score_disc
        for attr_no, attr in enumerate(domain.attributes):
            col_x = data.X[:, attr_no]
            if is_sparse:
                col_x = col_x.toarray()
                col_x = col_x.flatten()
            sc, *res = _score_cont() # disc_scorer() if attr.is_discrete else _score_cont()
            if res[0] is not None and sc > best_score:
                best_score, best_res = sc, res
        best_res[0].value = distribution.Discrete(data, class_var)
        return best_res

    def _build_tree(self, data, active_inst, level=1, finish=False):
        # https://stackoverflow.com/a/3844832
        from itertools import groupby
        
        def _all_equal(iterable):
            g = groupby(iterable)
            return next(g, True) and not next(g, False)

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
            if branches is not None and _all_equal(branches):
                node, branches, n_children = Node(None, None, distr), None, 0
        node.subset = active_inst
        
        if branches is not None: # len(branches) - self.min_samples_leaf > np.sum(branches) >= self.min_samples_leaf
            # if _all_equal(branches):
            #     node_copy = deepcopy(node)
            #     node_copy.children = []
            # else:
                node.children = [
                    self._build_tree(data, active_inst[branches == br], level + 1, finish=len(active_inst[branches == br])==len(branches))
                    for br in range(n_children)]
        return node

    def fit_storage(self, data):
        # print("fit_storage", data.domain)
        # if self.binarize and any(
        #         attr.is_discrete and len(attr.values) > self.MAX_BINARIZATION
        #         for attr in data.domain.attributes):
        #     # No fallback in the script; widgets can prevent this error
        #     # by providing a fallback and issue a warning about doing so
        #     raise ValueError("Exhaustive binarization does not handle "
        #                      "attributes with more than {} values".
        #                      format(self.MAX_BINARIZATION))

        active_inst = np.nonzero(~np.isnan(data.Y))[0].astype(np.int32) 
        root = self._build_tree(data, active_inst)
        if root is None:
            distr = distribution.Discrete(data, data.domain.class_var)
            if np.sum(distr) == 0:
                distr[:] = 1
            root = Node(None, 0, distr)
        root.subset = active_inst
        model = TreeModel(data, root)
        return model
