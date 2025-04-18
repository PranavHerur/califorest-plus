import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings


class STLBRF_OOB(ClassifierMixin, BaseEstimator):
    """
    Sequential, Threshold-based, Leave-Best Random Forest using OOB Error.

    Feature selection algorithm that works by sequentially reducing features
    based on importance, while maintaining performance threshold estimated
    using the Out-of-Bag (OOB) error.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        error_increment=0.01,  # Tolerance for OOB score drop
        elim_percent=0.1,  # Percentage of features to remove each iteration
        min_features=2,
        random_state=42,
        # n_splits is removed, OOB is used instead
        **rf_params,  # Allow passing other RF params like max_features, class_weight etc.
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.error_increment = error_increment
        self.elim_percent = elim_percent
        self.min_features = min_features
        self.random_state = random_state
        self.rf_params = rf_params

    def _create_rf(self):
        """Helper method to create RF instance with OOB score enabled."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            oob_score=True,  # Enable OOB score calculation
            n_jobs=-1,  # Use all available cores
            **self.rf_params,
        )

    def fit(self, X, y):
        # Input validation
        X, y = check_X_y(X, y)
        # Store initial number of features for potential validation in transform/predict
        # self.n_features_in_ = X.shape[1] # Standard sklearn practice

        # Start with all features
        feature_mask = np.arange(X.shape[1])
        # Use -inf because OOB score is accuracy-like (higher is better)
        self.best_score_ = -np.inf
        # Keep track of the feature set that yielded the best score
        self.selected_features_ = np.copy(feature_mask)

        # Sequential feature elimination loop
        while len(feature_mask) > self.min_features:
            X_current = X[:, feature_mask]
            current_rf = self._create_rf()

            # Fit the RF on the current feature subset and get OOB score
            try:
                current_rf.fit(X_current, y)
                if not hasattr(current_rf, "oob_score_") or np.isnan(
                    current_rf.oob_score_
                ):
                    warnings.warn(
                        f"OOB score not available or NaN for {len(feature_mask)} features. "
                        f"Ensure n_estimators ({self.n_estimators}) is large enough. Stopping elimination.",
                        UserWarning,
                    )
                    break
                current_score = current_rf.oob_score_
            except ValueError as e:
                warnings.warn(
                    f"Could not fit RF or compute OOB score with {len(feature_mask)} features: {e}. Stopping elimination.",
                    UserWarning,
                )
                break

            # Decision based on OOB score drop
            # If score drops below threshold compared to *best score found so far*
            if current_score < (self.best_score_ - self.error_increment):
                break  # Stop elimination, self.selected_features_ holds the best set

            # Update best score and selected features if current score is better or equal
            # Allows keeping fewer features if score is maintained
            best_rf_for_importances = None  # Initialize here
            if current_score >= self.best_score_:
                self.best_score_ = current_score
                self.selected_features_ = np.copy(feature_mask)
                # Store the RF that gave the best score, including its importances
                # We need this importance for feature removal
                best_rf_for_importances = current_rf
            # elif 'best_rf_for_importances' not in locals(): # Check if it was ever assigned
            #     # This case should ideally not happen if best_score_ starts at -inf
            #     # but as a safeguard, use the current RF if no better one was found yet
            #     best_rf_for_importances = current_rf
            if best_rf_for_importances is None:
                # If the first iteration already failed the check above, we still need importances
                best_rf_for_importances = current_rf

            # --- Feature Removal (using importances from the RF of the best scoring iteration) ---
            importances = best_rf_for_importances.feature_importances_

            # Calculate number of features to remove based on elim_percent
            num_features_current = len(feature_mask)
            num_to_remove = max(1, int(self.elim_percent * num_features_current))

            # Ensure we don't remove too many features and go below min_features
            if num_features_current - num_to_remove < self.min_features:
                # If removing the standard percentage drops below min_features,
                # only remove enough to reach min_features, unless we are already there.
                if num_features_current > self.min_features:
                    num_to_remove = num_features_current - self.min_features
                else:
                    break  # Already at or below min_features, stop.

            # Find indices of least important features *relative to the current feature_mask*
            least_important_relative_indices = np.argsort(importances)[:num_to_remove]

            # Remove the features from the mask
            feature_mask = np.delete(feature_mask, least_important_relative_indices)

        # Final training on the best selected features identified during the loop
        X_selected = X[:, self.selected_features_]
        self.rf_ = self._create_rf()  # Create final RF instance
        try:
            self.rf_.fit(X_selected, y)
            # Store the OOB score of the final model if available
            if hasattr(self.rf_, "oob_score_") and not np.isnan(self.rf_.oob_score_):
                self.final_oob_score_ = self.rf_.oob_score_
            else:
                self.final_oob_score_ = np.nan
        except ValueError as e:
            warnings.warn(f"Could not fit final RF model: {e}", UserWarning)
            # Consider how to handle failure here - maybe raise error or leave un-fitted?
            self.rf_ = None  # Indicate final fit failed
            self.final_oob_score_ = np.nan

        # Store number of features seen during fit
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = hasattr(self, "rf_") and self.rf_ is not None
        return self

    def transform(self, X):
        """Apply feature selection to X."""
        check_is_fitted(self, ["is_fitted_", "selected_features_", "n_features_in_"])
        if not self.is_fitted_:
            raise RuntimeError("This STLBRF_OOB instance is not fitted yet.")
        X = check_array(X, ensure_2d=True)
        # Check that the number of features matches the number seen during fit
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but STLBRF_OOB is expecting {self.n_features_in_} features as input."
            )
        return X[:, self.selected_features_]

    def fit_transform(self, X, y):
        """Fit to data, then transform it."""
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X):
        """Predict class for X using the final model trained on selected features."""
        check_is_fitted(self, ["is_fitted_", "rf_"])
        if not self.is_fitted_:
            raise RuntimeError("This STLBRF_OOB instance is not fitted yet.")
        # Transform X to select the features before predicting
        X_transformed = self.transform(X)
        return self.rf_.predict(X_transformed)

    def predict_proba(self, X):
        """Predict class probabilities for X using the final model."""
        check_is_fitted(self, ["is_fitted_", "rf_"])
        if not self.is_fitted_:
            raise RuntimeError("This STLBRF_OOB instance is not fitted yet.")
        # Transform X to select the features before predicting probabilities
        X_transformed = self.transform(X)
        return self.rf_.predict_proba(X_transformed)
