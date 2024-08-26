"""Tree learner widget"""

from collections import OrderedDict

from AnyQt.QtCore import Qt

from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.localization import pl
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview

from .util.uncertaintreelearner import UncertainTreeLearner

class OWUncertaintyTree(OWBaseLearner):
    """Tree algorithm with forward pruning."""
    name = "Uncertainty Tree"
    description = "A tree algorithm for data with uncertainties"
    icon = "icons/Tree.svg"
    priority = 30
    keywords = "tree, uncertainty tree"

    LEARNER = UncertainTreeLearner

    binary_trees = Setting(True)
    limit_min_leaf = Setting(True)
    min_leaf = Setting(2)
    limit_min_internal = Setting(True)
    min_internal = Setting(5)
    limit_depth = Setting(True)
    max_depth = Setting(100)
    uncertainty_multiplyer = Setting(.5)
    post_hoc = Setting(False)

    # Classification only settings
    limit_majority = Setting(True)
    sufficient_majority = Setting(95)

    spin_boxes = (
        ("Min. number of instances in leaves: ",
         "limit_min_leaf", "min_leaf", 1, 1000),
        ("Do not split subsets smaller than: ",
         "limit_min_internal", "min_internal", 1, 1000),
        ("Limit the maximal tree depth to: ",
         "limit_depth", "max_depth", 1, 1000),
        ("Stop when majority reaches [%]: ",
         "limit_majority", "sufficient_majority", 51, 100))
        
    doubleSpin_boxes = ("Uncertainty multiplyer: ", "uncertainty_multiplyer", 0, 2, 0.001)

    classification_spin_boxes = (
        ("Stop when majority reaches [%]: ",
         "limit_majority", "sufficient_majority", 51, 100),)

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, 'Uncertianty Parameters')
        # the checkbox is put into vBox for alignemnt with other checkboxes
        # gui.checkBox(box, self, "binary_trees", "Induce binary tree",
        #              callback=self.settings_changed,
        #              attribute=Qt.WA_LayoutUsesWidgetRect)
        label, setting, fromv, tov, incv = self.doubleSpin_boxes
        gui.doubleSpin(box, self, setting, fromv, tov, incv, label=label,
                 alignment=Qt.AlignRight,
                 callback=self.settings_changed,
                 controlWidth=80)
        gui.checkBox(box, self, "post_hoc", "Post hoc pruning",
                     callback=self.settings_changed,
                     attribute=Qt.WA_LayoutUsesWidgetRect)
        box = gui.widgetBox(self.controlArea, 'Parameters')
        for label, check, setting, fromv, tov in self.spin_boxes:
            gui.spin(box, self, setting, fromv, tov, label=label,
                     checked=check, alignment=Qt.AlignRight,
                     callback=self.settings_changed,
                     checkCallback=self.settings_changed, controlWidth=80)

    # def add_classification_layout(self, box):
    #     for label, check, setting, minv, maxv in self.classification_spin_boxes:
    #         gui.spin(box, self, setting, minv, maxv,
    #                  label=label, checked=check, alignment=Qt.AlignRight,
    #                  callback=self.settings_changed, controlWidth=80,
    #                  checkCallback=self.settings_changed)

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
                self.limit_majority],
            uncertainty_multiplyer=self.uncertainty_multiplyer,
            post_hoc=self.post_hoc)

    def create_learner(self):
        # pylint: disable=not-callable
        return self.LEARNER(**self.learner_kwargs())

    def get_learner_parameters(self):
        # from Orange.widgets.report import plural_w
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
    WidgetPreview(OWUncertaintyTree).run(Table("iris"))