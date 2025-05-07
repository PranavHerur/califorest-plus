import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from califorest.rfva import RFVA
from sklearn.ensemble import RandomForestClassifier
from run_chil_exp import read_data
from califorest import metrics as em


RANDOM_SEED = 42


def _random_forest_baseline(X_train, X_test, y_train, y_test):
    # Initialize and fit the model
    print("\nTraining RC30 model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate accuracy
    print(y_test.shape)
    print(y_pred.shape)
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print("\Random Forest Results on Iris Dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print some example predictions
    print("\nExample Predictions (first 5 samples):")
    for i in range(5):
        print(
            f"True label: {y_test[i]}, Predicted: {y_pred[i]}, Probabilities: {y_pred_proba[i]}"
        )

    y_pred_proba = model.predict_proba(X_test)
    _metrics(y_test, y_pred_proba[:, 1])


def _rfva(X_train, X_test, y_train, y_test):
    # Initialize and fit the model
    print("\nTraining RFVA model...")
    model = RFVA(n_estimators=100, max_depth=4)
    model.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate accuracy
    print(y_test.shape)
    print(y_pred.shape)
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print("\RFVA Results on Iris Dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Print some example predictions
    print("\nExample Predictions (first 5 samples):")
    for i in range(5):
        print(
            f"True label: {y_test[i]}, Predicted: {y_pred[i]}, Probabilities: {y_pred_proba[i]}"
        )

    y_pred_proba = model.predict_proba(X_test)
    _metrics(y_test, y_pred_proba[:, 1])


def _metrics(y_test, y_pred) -> dict:
    score_auc = roc_auc_score(y_test, y_pred)
    score_hl = em.hosmer_lemeshow(y_test, y_pred)
    score_sh = em.spiegelhalter(y_test, y_pred)
    score_b, score_bs = em.scaled_brier_score(y_test, y_pred)
    rel_small, rel_large = em.reliability(y_test, y_pred)

    results = {
        "random_seed": RANDOM_SEED,
        "auc": score_auc,
        "brier": score_b,
        "brier_scaled": score_bs,
        "hosmer_lemshow": score_hl,
        "speigelhalter": score_sh,
        "reliability_small": rel_small,
        "reliability_large": rel_large,
    }

    # Print results
    print(f"  AUC: {score_auc:.4f}")
    print(f"  Brier Score: {score_b:.4f}")
    print(f"  Scaled Brier Score: {score_bs:.4f}")
    print(f"  Hosmer-Lemeshow p-value: {score_hl:.4f}")
    print(f"  Spiegelhalter p-value: {score_sh:.4f}")
    print(f"  Reliability-in-the-small: {rel_small:.6f}")
    print(f"  Reliability-in-the-large: {rel_large:.6f}")
    print()

    return results


def main():
    mimic_size = "1000_subjects"
    X_train, X_test, y_train, y_test = read_data(
        "mimic3_mort_hosp", random_seed=RANDOM_SEED, mimic_size=mimic_size
    )

    # _rfva(X_train, X_test, y_train, y_test)
    # _random_forest_baseline(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
