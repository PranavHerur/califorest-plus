import os
from pprint import pprint
import time
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from analysis.run_chil_exp import read_data
from califorest.califorest import CaliForest
from califorest.rc30 import RC30
from califorest import metrics as em

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


n_estimators = 300
max_depth = 10
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


def _build_results_row(name, data):
    data["model"] = name
    return data


def plot_results(results_df, output_dir):
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
        "RC-Iso": "indianred",
        "RC-Logit": "orange",
        "RF-NoCal": "lightgray",
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
    # plt.show()
    plt.savefig(f"{output_dir}/boxplot.png")


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
    "RC-Iso": RC30(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ctype="isotonic",
    ),
    "RC-Logit": RC30(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        ctype="logistic",
    ),
    "RF-NoCal": RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    ),
}


datasets = [
    "mimic3_mort_hosp",
    "mimic3_mort_icu",
    "mimic3_los_3",
    "mimic3_los_7",
]
mimic_size = "full_mimic3"
EXPERIMENTS = 10

for dataset in datasets:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"califorest_original_results/{dataset}/{mimic_size}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = read_data(
        dataset, random_seed=random_seed, mimic_size=mimic_size
    )

    print(f"starting experiment for {dataset} with {mimic_size} subjects")

    results = []
    total_time = time.time()
    for exp_i in range(EXPERIMENTS):
        experiment_start_time = time.time()
        print(f"running experiment run {exp_i}")
        for name, model in models.items():
            print(name)
            start_time = time.time()
            metrics = _run_model(model, X_train, X_test, y_train, y_test)
            end_time = time.time()
            print(f"model {name} time taken: {end_time - start_time} seconds")
            print("*" * 5, end="\n\n")

            results.append(_build_results_row(name, metrics))
        experiment_end_time = time.time()
        print(
            f"experiment {exp_i} time taken: {experiment_end_time - experiment_start_time} seconds"
        )

    total_time = time.time() - total_time
    print(f"total time taken: {total_time} seconds")

    results_df = pd.DataFrame(results)
    plot_results(results_df, output_dir)

    results_df.to_csv(f"{output_dir}/{dataset}_{mimic_size}.csv", index=False)
