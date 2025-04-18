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

from run_chil_exp import read_data
from califorest.stlbrf import STLBRF
from califorest.stlbrf2 import STLBRF2


def _get_random_data():
    # Create a synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def main(X_train, X_test, y_train, y_test):
    # Test STLBRF model
    print("Testing STLBRF model...")
    stlbrf = STLBRF(
        n_estimators=100,
        error_increment=0.01,
        elim_percent=0.1,
        random_state=42,
        min_features=round(X_train.shape[1] * 0.5),
    )

    # Fit the model
    stlbrf.fit(X_train, y_train)

    # Get selected features
    print(
        f"Selected {len(stlbrf.selected_features_)} features: {stlbrf.selected_features_}"
    )

    # Test predictions
    y_pred = stlbrf.predict(X_test)
    y_proba = stlbrf.predict_proba(X_test)
    stlbrf_accuracy = accuracy_score(y_test, y_pred)
    stlbrf_auc = roc_auc_score(y_test, y_proba[:, 1])
    print(f"STLBRF Accuracy: {stlbrf_accuracy:.4f}")
    print(f"STLBRF AUC: {stlbrf_auc:.4f}")

    # Compare to standard Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba[:, 1])
    print(f"Standard RF Accuracy: {rf_accuracy:.4f}")
    print(f"Standard RF AUC: {rf_auc:.4f}")

    # Plot feature selection
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance and Selection")
    all_features = np.arange(X_train.shape[1])

    # Create feature importances array for all features
    all_importances = np.zeros(X_train.shape[1])
    # Map the selected feature importances to their original indices
    for i, feat_idx in enumerate(stlbrf.selected_features_):
        all_importances[feat_idx] = stlbrf.rf_.feature_importances_[i]

    # Plot bars
    bars = plt.bar(all_features, all_importances)
    for i in all_features:
        if i in stlbrf.selected_features_:
            bars[i].set_color("green")  # Selected features
        else:
            bars[i].set_color("red")  # Eliminated features

    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("stlbrf_feature_selection.png")
    print("Feature importance plot saved as 'stlbrf_feature_selection.png'")

    # Test probability predictions
    proba = stlbrf.predict_proba(X_test)
    print(f"Probability shape: {proba.shape}")
    print(f"Sample probabilities: {proba[0]}")


if __name__ == "__main__":
    # main(*_get_random_data())

    # time this execution
    start_time = time.time()
    mimic_size = "5000_subjects"
    dataset = "mimic3_mort_icu"
    main(*read_data(dataset, random_seed=42, mimic_size=mimic_size))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


# -------- STLBRF Test --------
def test_stlbrf(X_train, X_test, y_train, y_test):
    print("\n--- Testing STLBRF model ---")
    stlbrf = STLBRF(
        n_estimators=100,
        error_increment=0.01,
        elim_percent=0.1,
        random_state=42,
        min_features=max(
            2, round(X_train.shape[1] * 0.1)
        ),  # Ensure min_features is at least 2
    )
    stlbrf.fit(X_train, y_train)
    print(
        f"Selected {len(stlbrf.selected_features_)} features (STLBRF): {np.sort(stlbrf.selected_features_)}"
    )
    y_pred = stlbrf.predict(X_test)
    y_proba = stlbrf.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    print(f"STLBRF Accuracy: {accuracy:.4f}")
    print(f"STLBRF AUC: {auc:.4f}")
    return stlbrf


# -------- STLBRF2 Test --------
def test_stlbrf2(X_train, X_test, y_train, y_test):
    print("\n--- Testing STLBRF2 model ---")
    stlbrf2 = STLBRF2(
        n_estimators=100,
        error_increment=0.01,
        importance_percentile_threshold=0.25,  # Use percentile threshold
        random_state=42,
        min_features=max(
            2, round(X_train.shape[1] * 0.1)
        ),  # Ensure min_features is at least 2
    )
    stlbrf2.fit(X_train, y_train)
    print(
        f"Selected {len(stlbrf2.selected_features_)} features (STLBRF2): {np.sort(stlbrf2.selected_features_)}"
    )
    y_pred = stlbrf2.predict(X_test)
    y_proba = stlbrf2.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    print(f"STLBRF2 Accuracy: {accuracy:.4f}")
    print(f"STLBRF2 AUC: {auc:.4f}")
    return stlbrf2


# -------- Standard RF Test --------
def test_standard_rf(X_train, X_test, y_train, y_test):
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
    model, X_train, title_suffix="STLBRF", filename="stlbrf_feature_selection.png"
):
    plt.figure(figsize=(12, 7))
    plt.title(f"Feature Importance and Selection ({title_suffix})")
    all_features = np.arange(X_train.shape[1])

    # Create feature importances array for all features
    all_importances = np.zeros(X_train.shape[1])
    # Map the selected feature importances to their original indices
    # Note: model.rf_ contains the final RF trained only on selected features
    # Access importances from the *final* trained model
    if (
        hasattr(model, "selected_features_")
        and hasattr(model, "rf_")
        and hasattr(model.rf_, "feature_importances_")
    ):
        final_importances = model.rf_.feature_importances_
        if len(final_importances) == len(model.selected_features_):
            for i, feat_idx in enumerate(model.selected_features_):
                all_importances[feat_idx] = final_importances[i]
        else:
            print(
                "Warning: Mismatch between final importances and selected features count."
            )
            # Fallback or alternative needed? Maybe use importances from last iteration?
            # For now, just plot zeros for unselected features
            for i, feat_idx in enumerate(model.selected_features_):
                # Attempting to map if possible, otherwise leave as 0
                if i < len(final_importances):
                    all_importances[feat_idx] = final_importances[i]

    # Plot bars
    bars = plt.bar(
        all_features, all_importances, color="red"
    )  # Default to red (eliminated)
    if hasattr(model, "selected_features_"):
        for i in model.selected_features_:
            if i < len(bars):
                bars[i].set_color("green")  # Selected features

    plt.xlabel("Feature Index")
    plt.ylabel("Importance (from final model on selected features)")
    plt.xticks(all_features)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Feature importance plot saved as '{filename}'")


# -------- Main Execution --------
def main(X_train, X_test, y_train, y_test):

    # Test models
    stlbrf_model = test_stlbrf(X_train, X_test, y_train, y_test)
    stlbrf2_model = test_stlbrf2(X_train, X_test, y_train, y_test)
    rf_model = test_standard_rf(X_train, X_test, y_train, y_test)

    # Plot feature selection results
    plot_feature_selection(
        stlbrf_model,
        X_train,
        title_suffix="STLBRF (Elim Percent)",
        filename="stlbrf_elim_percent_features.png",
    )
    plot_feature_selection(
        stlbrf2_model,
        X_train,
        title_suffix="STLBRF2 (Percentile)",
        filename="stlbrf_percentile_features.png",
    )


if __name__ == "__main__":
    # Optional: Use random data for quick testing
    # print("Using random data for testing...")
    # main(*_get_random_data())

    # Use real data
    start_time = time.time()
    mimic_size = "5000_subjects"
    dataset = "mimic3_mort_icu"
    print(f"\nLoading data: {dataset} ({mimic_size})...")
    data = read_data(dataset, random_seed=42, mimic_size=mimic_size)
    print(f"Data loaded. Train shape: {data[0].shape}, Test shape: {data[1].shape}")

    main(*data)

    end_time = time.time()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
