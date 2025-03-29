import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score


class STLBRF:
    def __init__(
        self, error_increment=0.01, elim_percent=0.1, n_splits=5, min_features=2
    ):
        self.error_increment = error_increment
        self.elim_percent = elim_percent
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.min_features = min_features
        self.selected_features_ = None
        self.best_score_ = 0.0
        self.rf_ = RandomForestClassifier(random_state=42)

    def fit(self, X, y):
        feature_mask = np.arange(X.shape[1])  # Track feature indices
        self.best_score_ = 0.0  # Keep track of the highest seen accuracy

        while len(feature_mask) > self.min_features:
            fold_scores = []
            for train_idx, val_idx in self.kf.split(X):
                X_train, X_val = (
                    X[train_idx][:, feature_mask],
                    X[val_idx][:, feature_mask],
                )
                self.rf_.fit(X_train, y[train_idx])
                preds = self.rf_.predict(X_val)
                fold_scores.append(accuracy_score(y[val_idx], preds))

            mean_score = np.mean(fold_scores)

            # If performance drops significantly, stop removing features
            if mean_score < (self.best_score_ - self.error_increment):
                break

            # Update best score if improving
            self.best_score_ = max(self.best_score_, mean_score)

            # Get feature importances and eliminate the least important ones
            feature_importances = self.rf_.feature_importances_
            num_to_remove = max(1, int(self.elim_percent * len(feature_mask)))

            # Remove least important features
            least_important = np.argsort(feature_importances)[:num_to_remove]
            feature_mask = np.delete(feature_mask, least_important)

        self.selected_features_ = feature_mask  # Store selected features

        # Final fit on all data with selected features
        X_selected = X[:, self.selected_features_]
        self.rf_.fit(X_selected, y)

    def transform(self, X):
        """Apply feature selection on new data."""
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Make predictions using the selected features."""
        print(f"predicting on shape {X.shape}")
        return self.rf_.predict(X)


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    test_train_split = 0.8
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - test_train_split, random_state=42
    )

    stlbrf = STLBRF()
    stlbrf.fit(X_train, y_train)

    # test transform
    X_transformed = stlbrf.transform(X)
    print("hmm", X_transformed.shape)

    print(
        "stlbrf",
        accuracy_score(y_test, stlbrf.predict(stlbrf.transform(X_test))),
    )

    rf = RandomForestClassifier()
    rf.fit(X, y)
    print("rf", accuracy_score(y_test, rf.predict(X_test)))

    # harder dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_informative=15,
        n_redundant=5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - test_train_split, random_state=42
    )
    stlbrf = STLBRF()
    stlbrf.fit(X_train, y_train)
    print("stlbrf", accuracy_score(y_test, stlbrf.predict(stlbrf.transform(X_test))))

    rf = RandomForestClassifier()
    rf.fit(X, y)
    print("shape", X_test.shape)
    print("rf", accuracy_score(y_test, rf.predict(X_test)))
