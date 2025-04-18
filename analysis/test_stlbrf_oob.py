import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.metrics_utils import metrics
from run_chil_exp import read_data
from califorest.stlbrf_oob import STLBRF_OOB


def _get_random_data():
    """Creates a simple synthetic dataset for quick testing."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


# -------- STLBRF_OOB Test --------
def test_stlbrf_oob(X_train, X_test, y_train, y_test):
    """Tests the STLBRF_OOB model and prints performance metrics."""
    print("\n--- Testing STLBRF_OOB model ---")
    stlbrf_oob = STLBRF_OOB(
        n_estimators=100,
        error_increment=0.01,
        elim_percent=0.1,
        random_state=42,
        min_features=max(
            2, round(X_train.shape[1] * 0.1)
        ),  # Ensure min_features is at least 2
    )
    stlbrf_oob.fit(X_train, y_train)
    print(
        f"Selected {len(stlbrf_oob.selected_features_)} features (STLBRF_OOB): {np.sort(stlbrf_oob.selected_features_)}"
    )
    y_pred = stlbrf_oob.predict(X_test)
    y_proba = stlbrf_oob.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    metrics(y_test, y_proba[:, 1], 42)
    print(f"STLBRF_OOB Accuracy: {accuracy:.4f}")
    return stlbrf_oob


# -------- Standard RF Test (Optional Comparison) --------
def test_standard_rf(X_train, X_test, y_train, y_test):
    """Tests a standard Random Forest for comparison."""
    print("\n--- Comparing to Standard Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    print(f"Standard RF Accuracy: {accuracy:.4f}")
    print(f"Standard RF AUC: {auc:.4f}")
    return rf


# -------- Plotting Function --------
def plot_feature_selection(
    model,
    X_train,
    title_suffix="STLBRF_OOB",
    filename="stlbrf_oob_feature_selection.png",
):
    """Plots the feature importances and selection status."""
    plt.figure(figsize=(12, 7))
    plt.title(f"Feature Importance and Selection ({title_suffix})")
    all_features = np.arange(X_train.shape[1])

    # Create feature importances array for all features, initialized to zero
    all_importances = np.zeros(X_train.shape[1])

    # Map the selected feature importances to their original indices
    if (
        hasattr(model, "selected_features_")
        and hasattr(model, "rf_")
        and hasattr(model.rf_, "feature_importances_")
    ):
        final_importances = model.rf_.feature_importances_
        if len(final_importances) == len(model.selected_features_):
            for i, feat_idx in enumerate(model.selected_features_):
                if 0 <= feat_idx < len(all_importances):
                    all_importances[feat_idx] = final_importances[i]
        else:
            print(
                f"Warning: Mismatch between final importances ({len(final_importances)}) and selected features count ({len(model.selected_features_)}). Plotting may be incomplete."
            )
            for i, feat_idx in enumerate(model.selected_features_):
                if i < len(final_importances) and 0 <= feat_idx < len(all_importances):
                    all_importances[feat_idx] = final_importances[i]

    # Plot bars, default color red (eliminated)
    bars = plt.bar(all_features, all_importances, color="red")

    # Color selected features green
    if hasattr(model, "selected_features_"):
        for feat_idx in model.selected_features_:
            if 0 <= feat_idx < len(bars):
                bars[feat_idx].set_color("green")

    plt.xlabel("Feature Index")
    plt.ylabel("Importance (from final model on selected features)")
    plt.xticks(all_features)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Feature importance plot saved as '{filename}'")


# -------- Main Execution Logic --------
def run_tests(X_train, X_test, y_train, y_test):
    """Runs the tests and generates the plot for STLBRF_OOB."""
    # Test STLBRF_OOB model
    stlbrf_oob_model = test_stlbrf_oob(X_train, X_test, y_train, y_test)

    # Optionally, compare with standard RF
    rf_model = test_standard_rf(X_train, X_test, y_train, y_test)

    # Plot feature selection results for STLBRF_OOB
    plot_feature_selection(
        stlbrf_oob_model,
        X_train,
        title_suffix="STLBRF_OOB (OOB Error)",
        filename="stlbrf_oob_features.png",
    )


if __name__ == "__main__":
    # --- Configuration --- #
    USE_RANDOM_DATA = False  # Set to True to use synthetic data, False for MIMIC
    MIMIC_SIZE = "10000_subjects"
    DATASET = "mimic3_mort_icu"
    RANDOM_SEED = 42
    # --------------------- #

    print("--- STLBRF_OOB Test Script --- ")
    start_time = time.time()

    if USE_RANDOM_DATA:
        print("\nUsing random data for testing...")
        data = _get_random_data()
        print(
            f"Random data generated. Train shape: {data[0].shape}, Test shape: {data[1].shape}"
        )
    else:
        print(f"\nLoading data: {DATASET} ({MIMIC_SIZE})...")
        try:
            data = read_data(DATASET, random_seed=RANDOM_SEED, mimic_size=MIMIC_SIZE)
            print(
                f"Data loaded. Train shape: {data[0].shape}, Test shape: {data[1].shape}"
            )
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Please ensure the data path and `read_data` function are correct.")
            sys.exit(1)  # Exit if data loading fails

    # Run the tests using the loaded/generated data
    run_tests(*data)

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
