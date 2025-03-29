import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from califorest.rfva import RFVA
from sklearn.ensemble import RandomForestClassifier
from run_chil_exp import read_data


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


def main():
    X_train, X_test, y_train, y_test = read_data("breast_cancer", 42)

    _rfva(X_train, X_test, y_train, y_test)
    _random_forest_baseline(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
