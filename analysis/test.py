from pprint import pprint
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from venn_abers import VennAbersCalibrator

from califorest.califorest import CaliForest, ImprovedCaliForest
from califorest.rc30 import RC30
from run_chil_exp import read_data
from califorest import metrics as em

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

n_estimators = 300
max_depth = 8
min_samples_split = 3
min_samples_leaf = 1
random_seed = 42


def _metrics(y_test, y_pred) -> dict:
    score_auc = roc_auc_score(y_test, y_pred)
    score_hl = em.hosmer_lemeshow(y_test, y_pred)
    score_sh = em.spiegelhalter(y_test, y_pred)
    score_b, score_bs = em.scaled_brier_score(y_test, y_pred)
    rel_small, rel_large = em.reliability(y_test, y_pred)

    results = {
        "random_seed": random_seed,
        "auc": score_auc,
        "brier": score_b,
        "brier_scaled": score_bs,
        "hosmer_lemshow": score_hl,
        "speigelhalter": score_sh,
        "reliability_small": rel_small,
        "reliability_large": rel_large,
    }

    # Print results
    # print(f"  AUC: {score_auc:.4f}")
    # print(f"  Brier Score: {score_b:.4f}")
    # print(f"  Scaled Brier Score: {score_bs:.4f}")
    # print(f"  Hosmer-Lemeshow p-value: {score_hl:.4f}")
    # print(f"  Spiegelhalter p-value: {score_sh:.4f}")
    # print(f"  Reliability-in-the-small: {rel_small:.6f}")
    # print(f"  Reliability-in-the-large: {rel_large:.6f}")
    # print()

    return results


def _score_model(y_true, y_pred) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }

    # print(f"Accuracy: {metrics['accuracy']:.4f}")
    # print(f"Precision: {metrics['precision']:.4f}")
    # print(f"Recall: {metrics['recall']:.4f}")
    # print(f"F1 Score: {metrics['f1']:.4f}")
    # print("Confusion Matrix:")
    # print(metrics["confusion_matrix"])

    return metrics


def _run_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    # Generate probabilities and class predictions on the test set
    y_probs = model.predict_proba(X_test)
    y_pred = model.predict(X_test)

    # effectiveness of classifer
    print("regular:", str(model))
    results = {**_metrics(y_test, y_probs[:, 1]), **_score_model(y_test, y_pred)}
    return results


def _venn_abers(model, X_train, X_test, y_train, y_test):
    va = VennAbersCalibrator(
        estimator=model,
        inductive=False,
        cal_size=0.2,
        random_state=random_seed,
        n_splits=3,
        precision=4,
    )

    # Fit on the training set
    va.fit(X_train, y_train)

    # Generate probabilities and class predictions on the test set
    p_prime = va.predict_proba(X_test)
    y_pred = va.predict(X_test)

    # effectiveness of classifer
    print("with venn abers")
    results = {**_metrics(y_test, p_prime[:, 1]), **_score_model(y_test, y_pred[:, 1])}
    return results


def _build_results_row(name, data):
    data["model"] = name
    return data


def plot_results(results_df, dataset, mimic_size):
    """Create boxplots of results"""
    # Set style for plots
    sns.set_style("whitegrid")

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Define metrics to plot and their titles
    metrics = [
        ("brier_scaled", "Scaled Brier Score", axes[0, 0]),
        ("hosmer_lemshow", "Hosmer-Lemeshow p-value", axes[0, 1]),
        ("speigelhalter", "Spiegelhalter p-value", axes[0, 2]),
        ("brier", "Brier Score", axes[1, 0]),
        ("reliability_small", "Reliability-in-the-small", axes[1, 1]),
        ("reliability_large", "Reliability-in-the-large", axes[1, 2]),
    ]

    # Define colors for each model
    model_colors = {
        "CF-Iso": "skyblue",
        "CF-Logit": "lightgreen",
        "ImprovedCF-Iso": "purple",
        "ImprovedCF-logit": "orange",
        "RC-Iso": "indianred",
        "RF-NoCal": "lightgray",
        "CF-Iso-va": "skyblue",
        "CF-Logit-va": "lightgreen",
        "ImprovedCF-Iso-va": "purple",
        "ImprovedCF-logit-va": "orange",
        "RC-Iso-va": "indianred",
        "RF-NoCal-va": "lightgray",
    }

    # Plot each metric
    for metric_name, title, ax in metrics:
        # Create boxplot
        sns.boxplot(
            x="model",
            y=metric_name,
            hue="model",
            data=results_df,
            ax=ax,
            palette=model_colors,
            legend=False,
        )

        # Add title and set y-axis label
        ax.set_title(title)
        ax.set_ylabel("")

        # Remove x-axis label and rotate x-axis tick labels
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()
    plt.show()


models = {
    "CF-Iso": CaliForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ctype="isotonic",
    ),
    "CF-Logit": CaliForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ctype="logistic",
    ),
    "ImprovedCF-Iso": ImprovedCaliForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ctype="isotonic",
        ensemble_weight=0.7,  # Adjust weight to improve reliability
    ),
    "ImprovedCF-logit": ImprovedCaliForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ctype="logistic",
    ),
    "RC-Iso": RC30(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ctype="isotonic",
    ),
    "RF-NoCal": RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    ),
}

mimic_size = "5000_subjects"
dataset = "mimic3_mort_hosp"
EXPERIMENTS = 5

X_train, X_test, y_train, y_test = read_data(
    dataset, random_seed=random_seed, mimic_size=mimic_size
)

results = []
for exp_i in range(EXPERIMENTS):
    print(f"running experiment run {exp_i}")
    for name, model in models.items():
        print(name)
        vanila_results = _run_model(model, X_train, X_test, y_train, y_test)
        va_results = _venn_abers(model, X_train, X_test, y_train, y_test)
        print("*" * 5, end="\n\n")

        results.append(_build_results_row(name, vanila_results))
        results.append(_build_results_row(f"{name}-va", va_results))

results_df = pd.DataFrame(results)
plot_results(results_df, dataset, mimic_size)
