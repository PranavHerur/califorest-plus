from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np

threshold = 0.5


class STLBRF:
    def __init__(self, error_increment=0.05, elim_percent=0.1):
        self.base_rf = RandomForestClassifier()
        self.kf = KFold(n_splits=5)
        self.error_increment = error_increment
        self.elim_percent = elim_percent

    def fit(self, X, y):
        while self.error_increment < threshold:
            print(self.error_increment)
            fold_scores = []
            for train_idx, val_idx in self.kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                self.base_rf.fit(X_train, y[train_idx])
                fold_scores.append(self.base_rf.score(X_val, y[val_idx]))
            # Backward elimination logic here
            feature_importances = self.base_rf.feature_importances_
            sorted_indices = np.argsort(feature_importances)[::-1]
            sorted_importances = feature_importances[sorted_indices]
            cumsum_importances = np.cumsum(sorted_importances)
            num_features = len(cumsum_importances)
            num_features_to_elim = int(num_features * self.elim_percent)
            if num_features_to_elim > 0:
                # Eliminate the least important features
                X = X[:, sorted_indices[num_features_to_elim:]]
                self.base_rf.n_features_ = X.shape[1]

            # Update error increment
            error_increment = 1 - self.base_rf.score(X, y)
            self.error_increment = error_increment

        return self


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    X, y = load_iris(return_X_y=True)
    stlbrf = STLBRF()
    stlbrf.fit(X, y)
    print("fitted")
    # print(stlbrf.base_rf.feature_importances_)
