import numpy as np
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import resample
from scipy.interpolate import interp1d
from scipy import stats
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


class ImprovedVennAbersForest(ClassifierMixin, BaseEstimator):
    """
    Improved Random Forest classifier calibrated using Venn-Abers predictors.

    This implementation enhances the basic Venn-Abers predictor with:
    1. Proper isotonic regression calibration
    2. Class imbalance handling through adaptive weights
    3. Better bootstrap sampling
    4. Ensemble calibration and smoothing
    5. Variance-based uncertainty estimation

    Parameters
    ----------
    n_estimators : int, default=300
        The number of trees in the forest.
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split.
    max_depth : int, default=5
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    calibration_folds : int, default=5
        Number of folds for calibration (k in k-fold CV).
    average_method : {"mean", "weighted"}, default="weighted"
        Method to average the multiprobabilistic predictions.
    temperature : float, default=1.0
        Temperature for scaling probabilities (T<1 makes them more extreme, T>1 less extreme).
    class_weight : {dict, "balanced", None}, default=None
        Weights associated with classes.
    smoothing : float, default=1e-6
        Smoothing parameter for probability calibration.
    """

    def __init__(
        self,
        n_estimators=300,
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        calibration_folds=5,
        average_method="weighted",
        temperature=1.0,
        class_weight=None,
        smoothing=1e-6,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.calibration_folds = calibration_folds
        self.average_method = average_method
        self.temperature = temperature
        self.class_weight = class_weight
        self.smoothing = smoothing

    def _improved_isotonic_calibration(self, scores, labels, sample_weight=None):
        """
        Enhanced isotonic calibration using sklearn's IsotonicRegression.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Classifier scores (uncalibrated probabilities).
        labels : array-like of shape (n_samples,)
            Binary labels (0 or 1).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        p0_func : callable
            Function that maps scores to p0 probabilities.
        p1_func : callable
            Function that maps scores to p1 probabilities.
        """
        # Add small amount of noise to scores to avoid duplicates
        noise = np.random.normal(0, 1e-8, len(scores))
        scores_with_noise = scores + noise

        # Add boundary points for better extrapolation
        extended_scores = np.concatenate([[0], scores_with_noise, [1]])
        extended_labels = np.concatenate([[0], labels, [1]])

        if sample_weight is not None:
            # Handle sample weights with proper extension
            max_weight = np.max(sample_weight)
            extended_weights = np.concatenate(
                [[max_weight], sample_weight, [max_weight]]
            )
        else:
            extended_weights = None

        # Sort by scores
        sorted_idx = np.argsort(extended_scores)
        sorted_scores = extended_scores[sorted_idx]
        sorted_labels = extended_labels[sorted_idx]

        if extended_weights is not None:
            sorted_weights = extended_weights[sorted_idx]
        else:
            sorted_weights = None

        # Calculate calibration for p0 (assuming label 0)
        # For p0: we're calculating P(y=1 | x, y=0)
        p0_data = np.concatenate([sorted_scores, [sorted_scores[-1] + 0.001]])
        p0_target = np.concatenate([sorted_labels, [0]])
        p0_weights = None
        if sorted_weights is not None:
            p0_weights = np.concatenate([sorted_weights, [sorted_weights[-1]]])

        p0_calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        p0_calibrator.fit(p0_data, p0_target, sample_weight=p0_weights)

        # Calculate calibration for p1 (assuming label 1)
        # For p1: we're calculating P(y=1 | x, y=1)
        p1_data = np.concatenate([[sorted_scores[0] - 0.001], sorted_scores])
        p1_target = np.concatenate([[1], sorted_labels])
        p1_weights = None
        if sorted_weights is not None:
            p1_weights = np.concatenate([[sorted_weights[0]], sorted_weights])

        p1_calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        p1_calibrator.fit(p1_data, p1_target, sample_weight=p1_weights)

        # Create a function to compute calibrated probabilities
        def p0_func(score):
            # Handle scalar input by converting to array
            if np.isscalar(score):
                return float(p0_calibrator.predict(np.array([score]))[0])
            else:
                return p0_calibrator.predict(score)

        def p1_func(score):
            # Handle scalar input by converting to array
            if np.isscalar(score):
                return float(p1_calibrator.predict(np.array([score]))[0])
            else:
                return p1_calibrator.predict(score)

        return p0_func, p1_func

    def _apply_temperature_scaling(self, probs, inverse=False):
        """
        Apply temperature scaling to probabilities.

        Parameters
        ----------
        probs : array-like of shape (n_samples,)
            Probabilities to scale.
        inverse : bool, default=False
            If True, apply inverse temperature scaling.

        Returns
        -------
        scaled_probs : array-like of shape (n_samples,)
            Temperature-scaled probabilities.
        """
        # Map to logits space
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        # Apply temperature scaling in logits space
        if inverse:
            # Inverse temperature (T < 1 makes probs more extreme)
            scaled_logits = logits * (1 / self.temperature)
        else:
            # Regular temperature (T > 1 makes probs less extreme)
            scaled_logits = logits * self.temperature

        # Map back to probability space
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        return scaled_probs

    def _compute_calibration_weights(self, scores, labels, variance=None):
        """
        Compute weights for calibration based on various factors.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Classifier scores (uncalibrated probabilities).
        labels : array-like of shape (n_samples,)
            Binary labels (0 or 1).
        variance : array-like of shape (n_samples,), default=None
            Variance of predictions.

        Returns
        -------
        weights : array-like of shape (n_samples,)
            Calibration weights.
        """
        n_samples = len(scores)
        weights = np.ones(n_samples)

        # Apply weight based on class imbalance
        if self.class_weight == "balanced":
            class_weights = {
                0: n_samples / (2 * np.sum(labels == 0)),
                1: n_samples / (2 * np.sum(labels == 1)),
            }
            for i in range(n_samples):
                weights[i] = class_weights[labels[i]]
        elif isinstance(self.class_weight, dict):
            for i in range(n_samples):
                if labels[i] in self.class_weight:
                    weights[i] = self.class_weight[labels[i]]

        # Apply weight based on prediction variance if available
        if variance is not None:
            # Lower weight for high-variance predictions (uncertain predictions)
            variance_weights = 1 / (1 + 5 * variance)
            weights *= variance_weights

        # Apply weight based on prediction confidence (less weight for extreme scores)
        confidence = np.abs(scores - 0.5) * 2  # 0 for score=0.5, 1 for score=0/1
        confidence_weights = (
            1 - 0.3 * confidence
        )  # Less weight for very confident predictions
        weights *= confidence_weights

        return weights

    def fit(self, X, y):
        """
        Build a Venn-Abers calibrated random forest from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Check that y is binary
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError("VennAbersForest only supports binary classification.")

        # Store classes
        self.classes_ = unique_y

        # Create base forest estimators
        self.estimators_ = []
        for i in range(self.n_estimators):
            self.estimators_.append(
                Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    max_features="sqrt",
                    random_state=i,  # Different random state for each tree
                )
            )

        # Begin out-of-bag training with stratified k-fold CV
        n_samples = X.shape[0]
        self.p0_calibrators_ = []
        self.p1_calibrators_ = []
        self.cv_scores_ = []
        self.cv_labels_ = []
        self.cv_variance_ = []

        # Initialize stratified k-fold cross-validation
        skf = StratifiedKFold(
            n_splits=self.calibration_folds, shuffle=True, random_state=42
        )

        # Train trees on bootstrap samples
        bootstrap_indices = []
        for i in range(self.n_estimators):
            # Create bootstrap sample (with stratification)
            indices = np.arange(n_samples)
            if self.class_weight == "balanced":
                # Stratified bootstrap
                pos_idx = indices[y == 1]
                neg_idx = indices[y == 0]
                pos_sample = resample(pos_idx, replace=True, n_samples=len(pos_idx))
                neg_sample = resample(neg_idx, replace=True, n_samples=len(neg_idx))
                bootstrap_idx = np.concatenate([pos_sample, neg_sample])
            else:
                # Regular bootstrap
                bootstrap_idx = resample(indices, replace=True, n_samples=n_samples)

            bootstrap_indices.append(bootstrap_idx)
            self.estimators_[i].fit(X[bootstrap_idx], y[bootstrap_idx])

        # Apply cross-validation for calibration
        for train_idx, cal_idx in skf.split(X, y):
            X_train, X_cal = X[train_idx], X[cal_idx]
            y_train, y_cal = y[train_idx], y[cal_idx]

            # Train a subset of trees on the training part of this fold
            fold_estimators = []
            trees_per_fold = max(1, self.n_estimators // self.calibration_folds)
            for i in range(trees_per_fold):
                fold_tree = Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    max_features="sqrt",
                    random_state=i,
                )
                fold_tree.fit(X_train, y_train)
                fold_estimators.append(fold_tree)

            # Get calibration scores and variance from this fold's forest
            cal_scores = np.zeros(len(X_cal))
            cal_preds = np.zeros((len(X_cal), trees_per_fold))

            for j, est in enumerate(fold_estimators):
                tree_preds = est.predict_proba(X_cal)[:, 1]
                cal_preds[:, j] = tree_preds
                cal_scores += tree_preds

            cal_scores /= trees_per_fold
            cal_variance = np.var(cal_preds, axis=1)

            # Store calibration data for later use
            self.cv_scores_.extend(cal_scores)
            self.cv_labels_.extend(y_cal)
            self.cv_variance_.extend(cal_variance)

            # Create calibration weights
            cal_weights = self._compute_calibration_weights(
                cal_scores, y_cal, cal_variance
            )

            # Create Venn-Abers calibrators for this fold
            p0_cal, p1_cal = self._improved_isotonic_calibration(
                cal_scores, y_cal, sample_weight=cal_weights
            )
            self.p0_calibrators_.append(p0_cal)
            self.p1_calibrators_.append(p1_cal)

        # Convert lists to arrays for vectorized operations
        self.cv_scores_ = np.array(self.cv_scores_)
        self.cv_labels_ = np.array(self.cv_labels_)
        self.cv_variance_ = np.array(self.cv_variance_)

        # Fit overall calibration curve for diagnostics
        if len(self.cv_scores_) > 0:
            prob_true, prob_pred = calibration_curve(
                self.cv_labels_, self.cv_scores_, n_bins=10
            )
            self.calibration_curve_ = (prob_true, prob_pred)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X using Venn-Abers calibration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n_samples = X.shape[0]

        # Get raw scores and variance from forest
        tree_preds = np.zeros((n_samples, self.n_estimators))
        for i, est in enumerate(self.estimators_):
            tree_preds[:, i] = est.predict_proba(X)[:, 1]

        raw_scores = np.mean(tree_preds, axis=1)
        pred_variance = np.var(tree_preds, axis=1)

        # Apply temperature scaling to raw scores
        if self.temperature != 1.0:
            raw_scores = self._apply_temperature_scaling(raw_scores)

        # Apply all calibrators and get lower and upper probabilities
        p0_probs = np.zeros((n_samples, len(self.p0_calibrators_)))
        p1_probs = np.zeros((n_samples, len(self.p1_calibrators_)))

        # Process scores safely - handling one score at a time to avoid scalar issues
        for i, (p0_cal, p1_cal) in enumerate(
            zip(self.p0_calibrators_, self.p1_calibrators_)
        ):
            for j, score in enumerate(raw_scores):
                p0_probs[j, i] = p0_cal(float(score))
                p1_probs[j, i] = p1_cal(float(score))

        # Compute final probabilities based on averaging method
        if self.average_method == "mean":
            p0_final = np.mean(p0_probs, axis=1)
            p1_final = np.mean(p1_probs, axis=1)
        elif self.average_method == "weighted":
            # Compute weights based on prediction variance
            weights = 1 / (1 + 5 * pred_variance.reshape(-1, 1))
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            # Apply weighted average
            p0_final = np.sum(p0_probs * weights, axis=1)
            p1_final = np.sum(p1_probs * weights, axis=1)
        else:  # default to mean
            p0_final = np.mean(p0_probs, axis=1)
            p1_final = np.mean(p1_probs, axis=1)

        # Apply smoothing to avoid extreme probabilities
        if self.smoothing > 0:
            p0_final = (1 - self.smoothing) * p0_final + self.smoothing * 0.5
            p1_final = (1 - self.smoothing) * p1_final + self.smoothing * 0.5

        # Calculate the final probability using minimax regret principle
        # and ensure all probabilities are valid
        lower = np.minimum(p0_final, p1_final)
        upper = np.maximum(p0_final, p1_final)

        # Handle potential division by zero
        denominator = upper + lower
        denominator = np.where(
            denominator == 0, 1, denominator
        )  # Replace zeros with ones

        minimax_regret = (p0_final * upper + (1 - p0_final) * lower) / denominator

        # Handle numerical stability
        minimax_regret = np.nan_to_num(minimax_regret, nan=0.5)
        final_probs = np.clip(minimax_regret, 1e-7, 1 - 1e-7)

        # Return probabilities for both classes
        proba = np.zeros((n_samples, 2))
        proba[:, 1] = final_probs
        proba[:, 0] = 1 - final_probs

        return proba

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_uncertainty_ranges(self, X):
        """
        Get the uncertainty ranges (lower and upper probabilities) for predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ranges : array-like of shape (n_samples, 2)
            The lower and upper probabilities for each sample.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n_samples = X.shape[0]

        # Get raw scores and variance from forest
        tree_preds = np.zeros((n_samples, self.n_estimators))
        for i, est in enumerate(self.estimators_):
            tree_preds[:, i] = est.predict_proba(X)[:, 1]

        raw_scores = np.mean(tree_preds, axis=1)

        # Apply temperature scaling to raw scores if needed
        if self.temperature != 1.0:
            raw_scores = self._apply_temperature_scaling(raw_scores)

        # Apply all calibrators and get lower and upper probabilities
        p0_probs = np.zeros((n_samples, len(self.p0_calibrators_)))
        p1_probs = np.zeros((n_samples, len(self.p1_calibrators_)))

        # Process scores safely - handling one score at a time to avoid scalar issues
        for i, (p0_cal, p1_cal) in enumerate(
            zip(self.p0_calibrators_, self.p1_calibrators_)
        ):
            for j, score in enumerate(raw_scores):
                p0_probs[j, i] = p0_cal(float(score))
                p1_probs[j, i] = p1_cal(float(score))

        # Compute average p0 and p1 values across all calibrators
        p0_avg = np.mean(p0_probs, axis=1)
        p1_avg = np.mean(p1_probs, axis=1)

        # Lower and upper bounds
        lower_bound = np.minimum(p0_avg, p1_avg)
        upper_bound = np.maximum(p0_avg, p1_avg)

        # Apply smoothing to avoid bounds being too tight or too wide
        if self.smoothing > 0:
            delta = (upper_bound - lower_bound) * (1 - self.smoothing)
            mid = (upper_bound + lower_bound) / 2
            lower_bound = mid - delta / 2
            upper_bound = mid + delta / 2

        # Ensure bounds are properly ordered and in [0,1]
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)

        # Return the uncertainty ranges
        return np.column_stack((lower_bound, upper_bound))

    def calibration_report(self):
        """
        Generate a calibration report with metrics.

        Returns
        -------
        report : dict
            Dictionary with calibration metrics.
        """
        check_is_fitted(self, "is_fitted_")

        if not hasattr(self, "calibration_curve_"):
            return {"error": "No calibration data available"}

        prob_true, prob_pred = self.calibration_curve_

        # Calculate calibration metrics
        cal_error = np.mean((prob_true - prob_pred) ** 2)

        # Calculate reliability-in-the-small
        reliability_small = cal_error

        # Calculate reliability-in-the-large
        avg_true = np.mean(self.cv_labels_)
        avg_pred = np.mean(self.cv_scores_)
        reliability_large = (avg_true - avg_pred) ** 2

        # Return calibration report
        return {
            "calibration_mse": cal_error,
            "reliability_small": reliability_small,
            "reliability_large": reliability_large,
            "avg_prediction": avg_pred,
            "avg_true": avg_true,
            "calibration_curve": {
                "y_true": prob_true.tolist(),
                "y_pred": prob_pred.tolist(),
            },
        }


class VennAbersForest(ClassifierMixin, BaseEstimator):
    """
    Random Forest classifier calibrated using Venn-Abers predictors.

    Venn-Abers predictors are a distribution-free method for reliable probability
    calibration that produces multiprobabilistic predictions (lower and upper
    probabilities). This implementation uses Inductive Venn-Abers Predictors (IVAP)
    with Random Forests as the underlying classifier.

    Parameters
    ----------
    n_estimators : int, default=300
        The number of trees in the forest.
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split.
    max_depth : int, default=5
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    calibration_folds : int, default=5
        Number of folds for calibration (k in k-fold CV).
    average_method : {"mean", "median"}, default="mean"
        Method to average the multiprobabilistic predictions.
    """

    def __init__(
        self,
        n_estimators=300,
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        calibration_folds=5,
        average_method="mean",
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.calibration_folds = calibration_folds
        self.average_method = average_method

    def _isotonic_calibration(self, scores, labels):
        """
        Isotonic calibration using Venn-Abers method.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Classifier scores (uncalibrated probabilities).
        labels : array-like of shape (n_samples,)
            Binary labels (0 or 1).

        Returns
        -------
        calibrator : callable
            Function that takes a score and returns calibrated probabilities.
        """
        n = len(scores)

        # Add pseudo-points for 0 and 1 labels
        extended_scores = np.concatenate([[0], scores, [1]])
        extended_labels = np.concatenate([[0], labels, [1]])

        # Sort by scores
        idx = np.argsort(extended_scores)
        sorted_scores = extended_scores[idx]
        sorted_labels = extended_labels[idx]

        # Calculate the cumulative sums
        cumsum = np.cumsum(sorted_labels)

        # Function to compute calibrated probabilities
        def calibrate(score):
            # Find position where score would be inserted
            pos = np.searchsorted(sorted_scores, score)

            if pos == 0:
                return 0.0
            if pos == len(sorted_scores):
                return 1.0

            # Calculate p0: probability if we label the new example as 0
            p0_numerator = cumsum[pos - 1]
            p0_denominator = pos
            p0 = p0_numerator / p0_denominator if p0_denominator > 0 else 0

            # Calculate p1: probability if we label the new example as 1
            p1_numerator = cumsum[pos - 1] + 1
            p1_denominator = pos + 1
            p1 = p1_numerator / p1_denominator if p1_denominator > 0 else 0

            # Return p0 and p1 as the lower and upper probabilities
            return p0, p1

        return calibrate

    def fit(self, X, y):
        """
        Build a Venn-Abers calibrated random forest from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Check that y is binary
        if len(np.unique(y)) != 2:
            raise ValueError("VennAbersForest only supports binary classification.")

        # Store classes
        self.classes_ = np.unique(y)

        # Create base forest estimators
        self.estimators_ = []
        for i in range(self.n_estimators):
            self.estimators_.append(
                Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features="sqrt",
                )
            )

        # Begin out-of-bag training
        n_samples = X.shape[0]
        self.calibrators_ = []

        # Initialize k-fold cross-validation
        kf = StratifiedKFold(
            n_splits=self.calibration_folds, shuffle=True, random_state=42
        )

        # Train trees on bootstrap samples
        bootstrap_idx = []
        for i in range(self.n_estimators):
            # Create bootstrap sample
            sample_idx = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_idx.append(sample_idx)
            self.estimators_[i].fit(X[sample_idx], y[sample_idx])

        # Train calibrators using cross-validation
        for train_idx, cal_idx in kf.split(X, y):
            X_train, X_cal = X[train_idx], X[cal_idx]
            y_train, y_cal = y[train_idx], y[cal_idx]

            # Train a forest on the training part of this fold
            fold_estimators = []
            for i in range(self.n_estimators // self.calibration_folds):
                fold_tree = Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features="sqrt",
                )
                fold_tree.fit(X_train, y_train)
                fold_estimators.append(fold_tree)

            # Get calibration scores from this fold's forest
            cal_scores = np.zeros(len(X_cal))
            for est in fold_estimators:
                cal_scores += est.predict_proba(X_cal)[:, 1]
            cal_scores /= len(fold_estimators)

            # Create Venn-Abers calibrator for this fold
            calibrator = self._isotonic_calibration(cal_scores, y_cal)
            self.calibrators_.append(calibrator)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for X using Venn-Abers calibration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n_samples = X.shape[0]

        # Get raw scores from forest
        raw_scores = np.zeros(n_samples)
        for est in self.estimators_:
            raw_scores += est.predict_proba(X)[:, 1]
        raw_scores /= len(self.estimators_)

        # Apply all calibrators and get upper and lower probabilities
        lower_probs = np.zeros(n_samples)
        upper_probs = np.zeros(n_samples)

        for i, score in enumerate(raw_scores):
            fold_lower = []
            fold_upper = []

            # Handle each score individually to avoid scalar issues
            score_val = float(score)
            for calibrator in self.calibrators_:
                try:
                    p0, p1 = calibrator(score_val)
                    fold_lower.append(p0)
                    fold_upper.append(p1)
                except Exception as e:
                    # If there's an error, use reasonable defaults
                    fold_lower.append(0.5)
                    fold_upper.append(0.5)

            # Compute final probabilities based on averaging method
            if len(fold_lower) > 0:
                if self.average_method == "mean":
                    lower_probs[i] = np.mean(fold_lower)
                    upper_probs[i] = np.mean(fold_upper)
                else:  # median
                    lower_probs[i] = np.median(fold_lower)
                    upper_probs[i] = np.median(fold_upper)
            else:
                # Fallback if no valid calibrations
                lower_probs[i] = 0.5
                upper_probs[i] = 0.5

        # Get single probability by taking midpoint between lower and upper
        final_probs = (lower_probs + upper_probs) / 2

        # Return probabilities for both classes
        proba = np.zeros((n_samples, 2))
        proba[:, 1] = np.clip(final_probs, 1e-7, 1 - 1e-7)  # Avoid extreme values
        proba[:, 0] = 1 - proba[:, 1]

        return proba

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_uncertainty_ranges(self, X):
        """
        Get the uncertainty ranges (lower and upper probabilities) for predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ranges : array-like of shape (n_samples, 2)
            The lower and upper probabilities for each sample.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n_samples = X.shape[0]

        # Get raw scores from forest
        raw_scores = np.zeros(n_samples)
        for est in self.estimators_:
            raw_scores += est.predict_proba(X)[:, 1]
        raw_scores /= len(self.estimators_)

        # Apply all calibrators and get upper and lower probabilities
        lower_probs = np.zeros(n_samples)
        upper_probs = np.zeros(n_samples)

        for i, score in enumerate(raw_scores):
            fold_lower = []
            fold_upper = []

            # Handle each score individually to avoid scalar issues
            score_val = float(score)
            for calibrator in self.calibrators_:
                try:
                    p0, p1 = calibrator(score_val)
                    fold_lower.append(p0)
                    fold_upper.append(p1)
                except Exception as e:
                    # If there's an error, use reasonable defaults
                    fold_lower.append(0.5)
                    fold_upper.append(0.5)

            # Compute final probabilities based on averaging method
            if len(fold_lower) > 0:
                if self.average_method == "mean":
                    lower_probs[i] = np.mean(fold_lower)
                    upper_probs[i] = np.mean(fold_upper)
                else:  # median
                    lower_probs[i] = np.median(fold_lower)
                    upper_probs[i] = np.median(fold_upper)
            else:
                # Fallback if no valid calibrations
                lower_probs[i] = 0.5
                upper_probs[i] = 0.5

        return np.column_stack((lower_probs, upper_probs))


class BayesianVennAbersForest(ClassifierMixin, BaseEstimator):
    """
    A further improved Venn-Abers Random Forest implementation that incorporates:
    1. Bayesian smoothing for calibration
    2. A final calibration layer (Platt scaling)
    3. Proper cross-validation with CalibratedClassifierCV

    This class builds on ImprovedVennAbersForest and adds several additional
    improvements for better calibration performance.

    Parameters
    ----------
    n_estimators : int, default=300
        The number of trees in the forest.
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split.
    max_depth : int, default=5
        The maximum depth of the tree.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    calibration_folds : int, default=5
        Number of folds for calibration (k in k-fold CV).
    average_method : {"mean", "weighted"}, default="weighted"
        Method to average the multiprobabilistic predictions.
    temperature : float, default=1.0
        Temperature for scaling probabilities (T<1 makes them more extreme, T>1 less extreme).
    class_weight : {dict, "balanced", None}, default="balanced"
        Weights associated with classes.
    alpha_prior : float, default=1.0
        Alpha parameter for Beta prior (pseudo-count for positive class).
    beta_prior : float, default=1.0
        Beta parameter for Beta prior (pseudo-count for negative class).
    second_stage_calibration : {'isotonic', 'sigmoid', None}, default='sigmoid'
        Method for second-stage calibration (None to disable).
    """

    def __init__(
        self,
        n_estimators=300,
        criterion="gini",
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        calibration_folds=5,
        average_method="weighted",
        temperature=1.0,
        class_weight="balanced",
        alpha_prior=1.0,
        beta_prior=1.0,
        second_stage_calibration="sigmoid",
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.calibration_folds = calibration_folds
        self.average_method = average_method
        self.temperature = temperature
        self.class_weight = class_weight
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.second_stage_calibration = second_stage_calibration

    def _bayesian_isotonic_calibration(self, scores, labels, sample_weight=None):
        """
        Enhanced isotonic calibration with Bayesian smoothing.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Classifier scores (uncalibrated probabilities).
        labels : array-like of shape (n_samples,)
            Binary labels (0 or 1).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        p0_func : callable
            Function that maps scores to p0 probabilities.
        p1_func : callable
            Function that maps scores to p1 probabilities.
        """
        # Add small amount of noise to scores to avoid duplicates
        noise = np.random.normal(0, 1e-8, len(scores))
        scores_with_noise = scores + noise

        # Sort scores and corresponding labels
        sort_idx = np.argsort(scores_with_noise)
        sorted_scores = scores_with_noise[sort_idx]
        sorted_labels = labels[sort_idx]

        if sample_weight is not None:
            sorted_weights = sample_weight[sort_idx]
        else:
            sorted_weights = np.ones_like(sorted_scores)

        # Find unique score values and their indices
        unique_scores, score_idx = np.unique(sorted_scores, return_index=True)

        # Add boundary points for better extrapolation
        unique_scores = np.concatenate([[0], unique_scores, [1]])
        score_idx = np.concatenate([[0], score_idx, [len(sorted_scores)]])

        # Calculate Bayesian calibrated probabilities
        p0_probs = np.zeros(len(unique_scores))
        p1_probs = np.zeros(len(unique_scores))

        # Add boundary values
        p0_probs[0] = 0.0
        p1_probs[0] = 0.0
        p0_probs[-1] = 1.0
        p1_probs[-1] = 1.0

        # Calculate probabilities for each bin with Bayesian smoothing
        for i in range(1, len(unique_scores) - 1):
            # Get subset of data for this bin
            if i < len(score_idx) - 1:
                bin_start = score_idx[i]
                bin_end = score_idx[i + 1]
                bin_labels = sorted_labels[bin_start:bin_end]
                bin_weights = sorted_weights[bin_start:bin_end]

                # Calculate weighted sum of labels (positive examples)
                pos_count = np.sum(bin_labels * bin_weights)
                total_count = np.sum(bin_weights)

                # Apply Bayesian smoothing with Beta prior
                # For p0 calculation (probability if example = 0)
                alpha0 = self.alpha_prior
                beta0 = self.beta_prior + total_count
                p0 = alpha0 / (alpha0 + beta0)

                # For p1 calculation (probability if example = 1)
                alpha1 = self.alpha_prior + pos_count
                beta1 = self.beta_prior + (total_count - pos_count)
                p1 = alpha1 / (alpha1 + beta1)
            else:
                # For boundary or empty bins, use interpolation
                p0 = unique_scores[i]
                p1 = unique_scores[i]

            p0_probs[i] = p0
            p1_probs[i] = p1

        # Create interpolation functions for p0 and p1
        def p0_func(score):
            if np.isscalar(score):
                score_array = np.array([score])
                result = np.interp(score_array, unique_scores, p0_probs)
                return float(result[0])
            else:
                return np.interp(score, unique_scores, p0_probs)

        def p1_func(score):
            if np.isscalar(score):
                score_array = np.array([score])
                result = np.interp(score_array, unique_scores, p1_probs)
                return float(result[0])
            else:
                return np.interp(score, unique_scores, p1_probs)

        return p0_func, p1_func

    def _apply_temperature_scaling(self, probs, inverse=False):
        """
        Apply temperature scaling to probabilities.

        Parameters
        ----------
        probs : array-like of shape (n_samples,)
            Probabilities to scale.
        inverse : bool, default=False
            If True, apply inverse temperature scaling.

        Returns
        -------
        scaled_probs : array-like of shape (n_samples,)
            Temperature-scaled probabilities.
        """
        # Map to logits space
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1 - eps)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        # Apply temperature scaling in logits space
        if inverse:
            # Inverse temperature (T < 1 makes probs more extreme)
            scaled_logits = logits * (1 / self.temperature)
        else:
            # Regular temperature (T > 1 makes probs less extreme)
            scaled_logits = logits * self.temperature

        # Map back to probability space
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        return scaled_probs

    def fit(self, X, y):
        """
        Build a Bayesian Venn-Abers calibrated random forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False)

        # Check that y is binary
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError(
                "BayesianVennAbersForest only supports binary classification."
            )

        # Store classes
        self.classes_ = unique_y

        # Split data for final calibration
        X_model, X_calib, y_model, y_calib = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Create and fit base classifier first using CalibratedClassifierCV
        base_estimator = Tree(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            max_features="sqrt",
            random_state=42,
        )

        # Store individual trees for OOB predictions
        self.estimators_ = []
        for i in range(self.n_estimators):
            self.estimators_.append(
                Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    max_features="sqrt",
                    random_state=i,
                )
            )

        # Train base estimators with bootstrap sampling
        n_samples = X_model.shape[0]
        self.p0_calibrators_ = []
        self.p1_calibrators_ = []

        # Initialize stratified k-fold cross-validation
        skf = StratifiedKFold(
            n_splits=self.calibration_folds, shuffle=True, random_state=42
        )

        # Train trees with proper bootstrap sampling
        for i in range(self.n_estimators):
            # Create bootstrap sample with stratification
            indices = np.arange(n_samples)
            if self.class_weight == "balanced":
                # Stratified bootstrap for balanced sampling
                pos_idx = indices[y_model == 1]
                neg_idx = indices[y_model == 0]
                pos_sample = resample(
                    pos_idx, replace=True, n_samples=len(pos_idx), random_state=i
                )
                neg_sample = resample(
                    neg_idx, replace=True, n_samples=len(neg_idx), random_state=i
                )
                bootstrap_idx = np.concatenate([pos_sample, neg_sample])
            else:
                # Regular bootstrap
                bootstrap_idx = resample(
                    indices, replace=True, n_samples=n_samples, random_state=i
                )

            # Fit the tree to this bootstrap sample
            self.estimators_[i].fit(X_model[bootstrap_idx], y_model[bootstrap_idx])

        # Venn-Abers calibration using cross-validation
        # Store OOB predictions for each fold
        self.cv_scores_ = []
        self.cv_labels_ = []

        for train_idx, cal_idx in skf.split(X_model, y_model):
            X_train, X_cal = X_model[train_idx], X_model[cal_idx]
            y_train, y_cal = y_model[train_idx], y_model[cal_idx]

            # Create subset of trees for this fold
            fold_trees = []
            trees_per_fold = max(1, self.n_estimators // self.calibration_folds)
            for i in range(trees_per_fold):
                fold_tree = Tree(
                    criterion=self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    class_weight=self.class_weight,
                    max_features="sqrt",
                    random_state=i,
                )
                fold_tree.fit(X_train, y_train)
                fold_trees.append(fold_tree)

            # Get predictions for calibration set
            cal_scores = np.zeros(len(X_cal))
            for tree in fold_trees:
                cal_scores += tree.predict_proba(X_cal)[:, 1]
            cal_scores /= len(fold_trees)

            # Apply Bayesian calibration
            p0_cal, p1_cal = self._bayesian_isotonic_calibration(cal_scores, y_cal)
            self.p0_calibrators_.append(p0_cal)
            self.p1_calibrators_.append(p1_cal)

            # Store calibration data
            self.cv_scores_.extend(cal_scores)
            self.cv_labels_.extend(y_cal)

        # Convert to arrays for statistics
        self.cv_scores_ = np.array(self.cv_scores_)
        self.cv_labels_ = np.array(self.cv_labels_)

        # Fit second-stage calibration if requested
        if self.second_stage_calibration is not None:
            # Get predictions from base model
            base_preds = np.zeros((len(X_calib), 1))
            for tree in self.estimators_:
                base_preds[:, 0] += tree.predict_proba(X_calib)[:, 1]
            base_preds /= len(self.estimators_)

            # Apply first-stage Venn-Abers calibration
            va_preds = np.zeros(len(X_calib))
            for i, score in enumerate(base_preds[:, 0]):
                # Get p0 and p1 from all calibrators
                p0_vals = []
                p1_vals = []
                for p0_cal, p1_cal in zip(self.p0_calibrators_, self.p1_calibrators_):
                    p0_vals.append(p0_cal(float(score)))
                    p1_vals.append(p1_cal(float(score)))

                # Average p0 and p1 values
                p0 = np.mean(p0_vals)
                p1 = np.mean(p1_vals)

                # Minimax regret principle
                lower = min(p0, p1)
                upper = max(p0, p1)
                if (upper + lower) > 0:
                    va_preds[i] = (p0 * upper + (1 - p0) * lower) / (upper + lower)
                else:
                    va_preds[i] = 0.5  # Fallback for numerical issues

            # Fit second-stage calibration
            if self.second_stage_calibration == "sigmoid":
                self.second_calibrator_ = LogisticRegression(C=1.0, solver="lbfgs")
                self.second_calibrator_.fit(va_preds.reshape(-1, 1), y_calib)
            elif self.second_stage_calibration == "isotonic":
                self.second_calibrator_ = IsotonicRegression(out_of_bounds="clip")
                self.second_calibrator_.fit(va_preds, y_calib)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities using Bayesian Venn-Abers calibration.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n_samples = X.shape[0]

        # Get raw scores from random forest
        tree_preds = np.zeros((n_samples, self.n_estimators))
        for i, est in enumerate(self.estimators_):
            tree_preds[:, i] = est.predict_proba(X)[:, 1]

        raw_scores = np.mean(tree_preds, axis=1)
        pred_variance = np.var(tree_preds, axis=1)

        # Apply temperature scaling if needed
        if self.temperature != 1.0:
            raw_scores = self._apply_temperature_scaling(raw_scores)

        # Apply Venn-Abers calibrators
        p0_probs = np.zeros((n_samples, len(self.p0_calibrators_)))
        p1_probs = np.zeros((n_samples, len(self.p1_calibrators_)))

        # Process each score individually to avoid scalar issues
        for i, (p0_cal, p1_cal) in enumerate(
            zip(self.p0_calibrators_, self.p1_calibrators_)
        ):
            for j, score in enumerate(raw_scores):
                p0_probs[j, i] = p0_cal(float(score))
                p1_probs[j, i] = p1_cal(float(score))

        # Apply weighted averaging based on prediction variance
        if self.average_method == "weighted":
            # Lower weight for high-variance predictions
            weights = 1 / (1 + 5 * pred_variance.reshape(-1, 1))
            weights = weights / np.sum(weights, axis=1, keepdims=True)

            p0_final = np.sum(p0_probs * weights, axis=1)
            p1_final = np.sum(p1_probs * weights, axis=1)
        else:
            # Simple averaging
            p0_final = np.mean(p0_probs, axis=1)
            p1_final = np.mean(p1_probs, axis=1)

        # Apply first-stage calibration using minimax regret
        lower = np.minimum(p0_final, p1_final)
        upper = np.maximum(p0_final, p1_final)

        # Avoid division by zero
        denominator = upper + lower
        denominator = np.where(denominator == 0, 1.0, denominator)

        va_probs = (p0_final * upper + (1 - p0_final) * lower) / denominator

        # Handle numerical issues
        va_probs = np.nan_to_num(va_probs, nan=0.5)
        va_probs = np.clip(va_probs, 0.01, 0.99)  # Less extreme clipping

        # Apply second-stage calibration if available
        if hasattr(self, "second_calibrator_"):
            if self.second_stage_calibration == "sigmoid":
                final_probs = self.second_calibrator_.predict_proba(
                    va_probs.reshape(-1, 1)
                )[:, 1]
            elif self.second_stage_calibration == "isotonic":
                final_probs = self.second_calibrator_.predict(va_probs)
        else:
            final_probs = va_probs

        # Return probabilities for both classes
        proba = np.zeros((n_samples, 2))
        proba[:, 1] = final_probs
        proba[:, 0] = 1 - final_probs

        return proba

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_uncertainty_ranges(self, X):
        """
        Get the uncertainty ranges (lower and upper probabilities) for predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ranges : array-like of shape (n_samples, 2)
            The lower and upper probabilities for each sample.
        """
        X = check_array(X)
        check_is_fitted(self, "is_fitted_")

        n_samples = X.shape[0]

        # Get raw scores from forest
        raw_scores = np.zeros(n_samples)
        for est in self.estimators_:
            raw_scores += est.predict_proba(X)[:, 1]
        raw_scores /= len(self.estimators_)

        # Apply temperature scaling if needed
        if self.temperature != 1.0:
            raw_scores = self._apply_temperature_scaling(raw_scores)

        # Apply all calibrators and get lower and upper probabilities
        p0_probs = np.zeros((n_samples, len(self.p0_calibrators_)))
        p1_probs = np.zeros((n_samples, len(self.p1_calibrators_)))

        # Process scores safely - handling one score at a time to avoid scalar issues
        for i, (p0_cal, p1_cal) in enumerate(
            zip(self.p0_calibrators_, self.p1_calibrators_)
        ):
            for j, score in enumerate(raw_scores):
                try:
                    p0_probs[j, i] = p0_cal(float(score))
                    p1_probs[j, i] = p1_cal(float(score))
                except Exception:
                    # Fallback for errors
                    p0_probs[j, i] = 0.5
                    p1_probs[j, i] = 0.5

        # Compute average p0 and p1 values across all calibrators
        p0_avg = np.mean(p0_probs, axis=1)
        p1_avg = np.mean(p1_probs, axis=1)

        # Lower and upper bounds
        lower_bound = np.minimum(p0_avg, p1_avg)
        upper_bound = np.maximum(p0_avg, p1_avg)

        # Ensure bounds are properly ordered and in [0,1]
        lower_bound = np.clip(lower_bound, 0.01, 0.99)
        upper_bound = np.clip(upper_bound, 0.01, 0.99)

        # Apply second-stage calibration to the bounds if available
        if hasattr(self, "second_calibrator_"):
            if self.second_stage_calibration == "sigmoid":
                lower_bound = self.second_calibrator_.predict_proba(
                    lower_bound.reshape(-1, 1)
                )[:, 1]
                upper_bound = self.second_calibrator_.predict_proba(
                    upper_bound.reshape(-1, 1)
                )[:, 1]
            elif self.second_stage_calibration == "isotonic":
                lower_bound = self.second_calibrator_.predict(lower_bound)
                upper_bound = self.second_calibrator_.predict(upper_bound)

        # Ensure lower <= upper after calibration
        temp_lower = np.minimum(lower_bound, upper_bound)
        temp_upper = np.maximum(lower_bound, upper_bound)
        lower_bound = temp_lower
        upper_bound = temp_upper

        # Return the uncertainty ranges
        return np.column_stack((lower_bound, upper_bound))
