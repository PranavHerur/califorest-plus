import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.isotonic import IsotonicRegression
import warnings

class VennAbersOOB(ClassifierMixin, BaseEstimator):
    """
    Random Forest classifier calibrated using Venn-Abers predictors on Out-of-Bag samples.
    
    This combines the strengths of Out-of-Bag estimation (efficient use of training data)
    with Venn-Abers calibration (distribution-free probability calibration).
    
    Parameters
    ----------
    n_estimators : int, default=300
        The number of trees in the forest.
    max_depth : int, default=None
        The maximum depth of the trees.
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node.
    class_weight : dict, 'balanced' or None, default=None
        Weights associated with classes. If None, all classes are equally weighted.
    random_state : int, RandomState instance or None, default=42
        Controls both the randomness of the bootstrapping and the sampling of features.
    """
    
    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",  # Standard RF default
        class_weight=None,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
    
    def _isotonic_calibration(self, scores, labels):
        """
        Apply isotonic calibration using Venn-Abers method.
        
        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Classifier scores (uncalibrated probabilities).
        labels : array-like of shape (n_samples,)
            Binary labels (0 or 1).
        
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
        
        # Sort by scores
        sorted_idx = np.argsort(extended_scores)
        sorted_scores = extended_scores[sorted_idx]
        sorted_labels = extended_labels[sorted_idx]
        
        # Calculate calibration for p0 (assuming label 0)
        p0_data = np.concatenate([sorted_scores, [sorted_scores[-1] + 0.001]])
        p0_target = np.concatenate([sorted_labels, [0]])
        
        p0_calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        p0_calibrator.fit(p0_data, p0_target)
        
        # Calculate calibration for p1 (assuming label 1)
        p1_data = np.concatenate([[sorted_scores[0] - 0.001], sorted_scores])
        p1_target = np.concatenate([[1], sorted_labels])
        
        p1_calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        p1_calibrator.fit(p1_data, p1_target)
        
        # Create functions to compute calibrated probabilities
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
        # Validate inputs
        X, y = check_X_y(X, y, accept_sparse=False)
        
        # Check that y is binary
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError("VennAbersOOB only supports binary classification.")
        
        # Store classes
        self.classes_ = unique_y
        
        # Store dimensions
        n_samples = X.shape[0]
        
        # Set random state
        rng = np.random.RandomState(self.random_state)
        
        # Create individual trees
        self.estimators_ = []
        
        # To collect OOB predictions
        oob_predictions = np.zeros((n_samples, len(unique_y)))
        oob_counts = np.zeros(n_samples)
        
        # Train trees with bootstrap sampling and collect OOB predictions
        for i in range(self.n_estimators):
            # Create bootstrap sample
            bootstrap_indices = rng.choice(n_samples, n_samples, replace=True)
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            
            # Create and train a decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=rng.randint(0, 2**31 - 1)  # Different for each tree
            )
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            self.estimators_.append(tree)
            
            # Get predictions for OOB samples
            if len(oob_indices) > 0:
                tree_preds = tree.predict_proba(X[oob_indices])
                oob_predictions[oob_indices] += tree_preds
                oob_counts[oob_indices] += 1
        
        # Average the OOB predictions
        oob_mask = oob_counts > 0
        for i in np.where(oob_mask)[0]:
            oob_predictions[i] /= oob_counts[i]
        
        # Handle samples that are never OOB (rare but possible)
        never_oob = oob_counts == 0
        if np.any(never_oob):
            warnings.warn(
                f"{np.sum(never_oob)} samples were never out-of-bag. "
                f"Increase n_estimators to reduce this number.",
                UserWarning,
            )
            # Use in-bag predictions for these (not ideal but a fallback)
            for i in np.where(never_oob)[0]:
                tree_preds = np.array([tree.predict_proba([X[i]])[0] for tree in self.estimators_])
                oob_predictions[i] = np.mean(tree_preds, axis=0)
        
        # Get OOB probabilities for the positive class
        oob_proba = oob_predictions[:, 1]
        
        # Apply Venn-Abers calibration on the OOB predictions
        self.p0_calibrator_, self.p1_calibrator_ = self._isotonic_calibration(oob_proba, y)
        
        # Store the OOB score (accuracy) for reference
        oob_predictions_class = np.argmax(oob_predictions, axis=1)
        self.oob_score_ = np.mean(oob_predictions_class == y)
        
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
        
        # Get raw scores from forest (average of tree predictions)
        raw_scores = np.zeros(n_samples)
        for est in self.estimators_:
            raw_scores += est.predict_proba(X)[:, 1]
        raw_scores /= len(self.estimators_)
        
        # Apply Venn-Abers calibration
        p0_probs = np.zeros(n_samples)
        p1_probs = np.zeros(n_samples)
        
        # Process scores safely - handling one score at a time to avoid scalar issues
        for i, score in enumerate(raw_scores):
            p0_probs[i] = self.p0_calibrator_(float(score))
            p1_probs[i] = self.p1_calibrator_(float(score))
        
        # Calculate the final probability using minimax regret principle
        lower = np.minimum(p0_probs, p1_probs)
        upper = np.maximum(p0_probs, p1_probs)
        
        # Handle potential division by zero
        denominator = upper + lower
        denominator = np.where(denominator == 0, 1, denominator)  # Replace zeros with ones
        
        minimax_regret = (p0_probs * upper + (1 - p0_probs) * lower) / denominator
        
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
        
        # Get raw scores from forest
        raw_scores = np.zeros(n_samples)
        for est in self.estimators_:
            raw_scores += est.predict_proba(X)[:, 1]
        raw_scores /= len(self.estimators_)
        
        # Apply Venn-Abers calibration
        p0_probs = np.zeros(n_samples)
        p1_probs = np.zeros(n_samples)
        
        # Process scores safely - handling one score at a time to avoid scalar issues
        for i, score in enumerate(raw_scores):
            p0_probs[i] = self.p0_calibrator_(float(score))
            p1_probs[i] = self.p1_calibrator_(float(score))
        
        # Lower and upper bounds
        lower_bound = np.minimum(p0_probs, p1_probs)
        upper_bound = np.maximum(p0_probs, p1_probs)
        
        # Ensure bounds are properly ordered and in [0,1]
        lower_bound = np.clip(lower_bound, 0, 1)
        upper_bound = np.clip(upper_bound, 0, 1)
        
        # Return the uncertainty ranges
        return np.column_stack((lower_bound, upper_bound))