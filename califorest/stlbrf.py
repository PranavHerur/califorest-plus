import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class STLBRF(ClassifierMixin, BaseEstimator):
    """
    Sequential, Threshold-based, Leave-Best Random Forest

    Feature selection algorithm that works by sequentially reducing features
    while maintaining performance threshold.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        error_increment=0.01,
        elim_percent=0.1,
        n_splits=5,
        min_features=2,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.error_increment = error_increment
        self.elim_percent = elim_percent
        self.n_splits = n_splits
        self.min_features = min_features
        self.random_state = random_state

    def fit(self, X, y):
        # Input validation
        X, y = check_X_y(X, y)

        # Initialize model and k-fold cross-validation
        self.rf_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.kf_ = KFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        # Start with all features
        feature_mask = np.arange(X.shape[1])
        self.best_score_ = 0.0

        # Sequential feature elimination
        while len(feature_mask) > self.min_features:
            fold_scores = []
            for train_idx, val_idx in self.kf_.split(X):
                X_train, X_val = (
                    X[train_idx][:, feature_mask],
                    X[val_idx][:, feature_mask],
                )
                self.rf_.fit(X_train, y[train_idx])
                preds = self.rf_.predict(X_val)
                fold_scores.append(accuracy_score(y[val_idx], preds))

            mean_score = np.mean(fold_scores)

            # If performance drops below threshold, stop
            if mean_score < (self.best_score_ - self.error_increment):
                break

            # Update best score if current is better
            self.best_score_ = max(self.best_score_, mean_score)

            # Calculate feature importances and remove least important ones
            importances = self.rf_.feature_importances_
            num_to_remove = max(1, int(self.elim_percent * len(feature_mask)))

            # Remove the least important features
            least_important = np.argsort(importances)[:num_to_remove]
            feature_mask = np.delete(feature_mask, least_important)

        # Store selected features
        self.selected_features_ = feature_mask

        # Final training on selected features with all data
        X_selected = X[:, self.selected_features_]
        self.rf_.fit(X_selected, y)

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Apply feature selection to X."""
        check_is_fitted(self, ["is_fitted_", "selected_features_"])
        X = check_array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Predict class for X."""
        check_is_fitted(self, ["is_fitted_", "selected_features_"])
        X_transformed = (
            self.transform(X) if X.shape[1] != len(self.selected_features_) else X
        )
        return self.rf_.predict(X_transformed)

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        check_is_fitted(self, ["is_fitted_", "selected_features_"])
        X_transformed = (
            self.transform(X) if X.shape[1] != len(self.selected_features_) else X
        )
        return self.rf_.predict_proba(X_transformed)
