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
