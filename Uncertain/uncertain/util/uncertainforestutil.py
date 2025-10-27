import numpy as np

from Orange.base import RandomForestModel
from Orange.data import Domain
from Orange.data.table import Table

from .uncertaintreelearner import UncertainTreeLearner

from Orange.classification import Learner

class UncertainForestModel(RandomForestModel):
        
    def __init__(self, trees, domain = None):
        super().__init__(domain)
        self.trees_arr = trees
    
    @property
    def trees(self):
        def wrap(tree, i):
            t = tree
            t.domain = self.domain
            t.supports_multiclass = self.supports_multiclass
            t.name = "{} - tree {}".format(self.name, i)
            t.original_domain = self.original_domain
            if hasattr(self, 'instances'):
                t.instances = self.instances
            return t

        return [wrap(tree, i)
                for i, tree in enumerate(self.trees_arr)]
        
    def predict(self, X):
        # Get shape of predictions
        predictions = self.trees_arr[0]._prepare_predictions(len(X)) 
        predictions[:,:] = 0
        
        for tree in self.trees_arr:
            predictions = np.add(predictions, tree.predict(X))
            
        # print(predictions / len(self.trees_arr))
        # print("len(self.trees_arr)",len(self.trees_arr))
        
        return predictions / len(self.trees_arr)

class UncertainForestLearner(Learner):
    __returns__ = UncertainForestModel
    supports_weights = True

    def __init__(self, n_trees=10, max_features=None, random_state=None,
                 max_depth=None, min_samples_split=2, preprocessors=None,
                 balanced_classes = False, uncertainty_multiplyer = 0.5, post_hoc = False, **kwargs):
        super().__init__(preprocessors=preprocessors)
        self.n_trees = n_trees
        self.max_features = max_features
        np.random.seed(random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.balanced_classes = balanced_classes
        self.uncertainty_multiplyer = uncertainty_multiplyer
        self.post_hoc = post_hoc
    
    def _bag_data(self, data):
        if self.max_features is not None:
            _ , features = data.X.shape
            if self.max_features < features:
                selected_columns = np.random.choice(features, size=self.max_features, replace=False)
                data = Table.from_numpy(Domain(attributes = np.array(data.domain.attributes)[selected_columns],
                                               class_vars = data.domain.class_var,
                                               metas      = data.domain.metas),
                                        X     = data.X[:, selected_columns],
                                        Y     = data.Y,
                                        metas = data.metas)
                
        if self.balanced_classes and data.domain.has_discrete_class:
            for i, value in enumerate(np.unique(data.Y)):
                indices, = np.where(data.Y == value)
                data_per_class = data[indices]
                bagged_data_class = data_per_class[np.random.choice(len(data_per_class), size=len(data_per_class), replace=True)]
                if i == 0:
                    bagged_data = bagged_data_class
                else:
                    bagged_data = Table.concatenate((bagged_data, bagged_data_class))
            return bagged_data
        
        bagged_data = data[np.random.choice(len(data), size=len(data), replace=True)]
        return bagged_data
    
    def fit_storage(self, data):
        trees = []
        for _ in range(self.n_trees):
            tree = UncertainTreeLearner(max_depth = self.max_depth,
                                        min_samples_split = self.min_samples_split,
                                        uncertainty_multiplyer = self.uncertainty_multiplyer,
                                        post_hoc = self.post_hoc) # TODO change parameters
            bagged_data = self._bag_data(data)
            model = tree.fit_storage(data=bagged_data)
            trees.append(model) 
        
        return UncertainForestModel(trees=trees)
