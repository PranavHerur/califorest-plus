from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from venn_abers import VennAbersCalibrator


class VennAbersForest(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_estimators=300,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
        )
        vennabers = VennAbersCalibrator(
            estimator=clf, inductive=False, n_splits=3, precision=4
        )
        self.vennabers = vennabers

    def fit(self, X, y):
        self.vennabers.fit(X, y)

    def predict(self, X):
        return self.vennabers.predict(X)

    def predict_proba(self, X):
        return self.vennabers.predict_proba(X)
