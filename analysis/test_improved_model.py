#!/usr/bin/env python3
import argparse
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

from run_chil_exp import read_data
from califorest import (
    CaliForest,
    RC30,
    ImprovedCaliForest,
    VennAbersForest,
    ImprovedVennAbersForest,
    BayesianVennAbersForest,
)
from califorest import metrics as em

# Define constants
n_estimators = 300
max_depth = 5
min_samples_split = 3
min_samples_leaf = 1
random_seed = 42


def run_experiment(dataset, random_seed=42, mimic_size="1000_subjects"):
    """Run experiment on the dataset with different models"""

    X_train, X_test, y_train, y_test = read_data(
        dataset, random_seed=random_seed, mimic_size=mimic_size
    )

    # Initialize models
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
        "VennAbers-mean": VennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            calibration_folds=5,
            average_method="mean",
        ),
        "VennAbers-median": VennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            calibration_folds=5,
            average_method="median",
        ),
        "ImprovedVA-bal": ImprovedVennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            calibration_folds=5,
            class_weight="balanced",
            temperature=1.1,
            average_method="weighted",
            smoothing=1e-4,
        ),
        "ImprovedVA-temp": ImprovedVennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            calibration_folds=5,
            temperature=0.9,  # Makes predictions more extreme
            average_method="weighted",
            smoothing=1e-4,
        ),
        "BayesVA-sigmoid": BayesianVennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            calibration_folds=5,
            alpha_prior=2.0,
            beta_prior=2.0,
            second_stage_calibration="sigmoid",
        ),
        "BayesVA-isotonic": BayesianVennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            calibration_folds=5,
            alpha_prior=2.0,
            beta_prior=2.0,
            second_stage_calibration="isotonic",
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

    # Store results
    results = []

    # Test each model
    for name, model in models.items():
        # print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        score_auc = roc_auc_score(y_test, y_pred)
        score_hl = em.hosmer_lemeshow(y_test, y_pred)
        score_sh = em.spiegelhalter(y_test, y_pred)
        score_b, score_bs = em.scaled_brier_score(y_test, y_pred)
        rel_small, rel_large = em.reliability(y_test, y_pred)

        # Store results
        results.append(
            {
                "model": name,
                "random_seed": random_seed,
                "auc": score_auc,
                "brier": score_b,
                "brier_scaled": score_bs,
                "hosmer_lemshow": score_hl,
                "speigelhalter": score_sh,
                "reliability_small": rel_small,
                "reliability_large": rel_large,
            }
        )

        # Print results
        # print(f"  AUC: {score_auc:.4f}")
        # print(f"  Brier Score: {score_b:.4f}")
        # print(f"  Scaled Brier Score: {score_bs:.4f}")
        # print(f"  Hosmer-Lemeshow p-value: {score_hl:.4f}")
        # print(f"  Spiegelhalter p-value: {score_sh:.4f}")
        # print(f"  Reliability-in-the-small: {rel_small:.6f}")
        # print(f"  Reliability-in-the-large: {rel_large:.6f}")
        # print()

    return pd.DataFrame(results)


def run_multiple_seeds(dataset, n_seeds=5, mimic_size="1000_subjects"):
    """Run experiment with multiple random seeds"""
    all_results = []

    for seed in range(n_seeds):
        print(f"Running with seed {seed}...")
        seed_results = run_experiment(dataset, random_seed=seed, mimic_size=mimic_size)
        all_results.append(seed_results)

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save results
    results_df.to_csv("my_results/improved_model_results.csv", index=False)

    return results_df


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
        "VennAbers-mean": "darkblue",
        "VennAbers-median": "steelblue",
        "ImprovedVA-bal": "teal",
        "ImprovedVA-temp": "navy",
        "BayesVA-sigmoid": "mediumseagreen",
        "BayesVA-isotonic": "darkolivegreen",
        "RC-Iso": "indianred",
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
    plt.show()

    # Save the figure
    # plt.savefig(
    #     f"my_results/{mimic_size}/{dataset}_improved_model_results.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )


def plot_uncertainty_ranges(dataset, X_test, y_test, mimic_size):
    """Create plots showing the uncertainty ranges from Venn-Abers models"""
    # Initialize models
    venn_abers_mean = VennAbersForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        calibration_folds=5,
        average_method="mean",
    )

    venn_abers_median = VennAbersForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        calibration_folds=5,
        average_method="median",
    )

    improved_va_balanced = ImprovedVennAbersForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        calibration_folds=5,
        class_weight="balanced",
        temperature=1.1,
        average_method="weighted",
        smoothing=1e-4,
    )

    improved_va_temp = ImprovedVennAbersForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        calibration_folds=5,
        temperature=0.9,  # Makes predictions more extreme
        average_method="weighted",
        smoothing=1e-4,
    )

    # Load data
    X_train, X_test, y_train, y_test = read_data(
        dataset, random_seed=random_seed, mimic_size=mimic_size
    )

    # Train models
    venn_abers_mean.fit(X_train, y_train)
    venn_abers_median.fit(X_train, y_train)
    improved_va_balanced.fit(X_train, y_train)
    improved_va_temp.fit(X_train, y_train)

    # Get predictions and uncertainty ranges
    pred_mean = venn_abers_mean.predict_proba(X_test)[:, 1]
    uncertainty_mean = venn_abers_mean.get_uncertainty_ranges(X_test)

    pred_median = venn_abers_median.predict_proba(X_test)[:, 1]
    uncertainty_median = venn_abers_median.get_uncertainty_ranges(X_test)

    pred_improved_bal = improved_va_balanced.predict_proba(X_test)[:, 1]
    uncertainty_improved_bal = improved_va_balanced.get_uncertainty_ranges(X_test)

    pred_improved_temp = improved_va_temp.predict_proba(X_test)[:, 1]
    uncertainty_improved_temp = improved_va_temp.get_uncertainty_ranges(X_test)

    # Create first figure for original Venn-Abers models
    fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))

    # Sort by prediction for better visualization
    mean_idx = np.argsort(pred_mean)
    median_idx = np.argsort(pred_median)

    # Plot 1: Uncertainty ranges for mean method
    axes1[0, 0].fill_between(
        range(len(pred_mean)),
        uncertainty_mean[mean_idx, 0],
        uncertainty_mean[mean_idx, 1],
        alpha=0.3,
        color="darkblue",
        label="Uncertainty Range",
    )
    axes1[0, 0].scatter(
        range(len(pred_mean)),
        pred_mean[mean_idx],
        s=10,
        color="darkblue",
        alpha=0.7,
        label="Prediction",
    )
    axes1[0, 0].set_title("VennAbers-mean: Predictions with Uncertainty Ranges")
    axes1[0, 0].set_xlabel("Samples (sorted by prediction)")
    axes1[0, 0].set_ylabel("Probability")
    axes1[0, 0].legend()
    axes1[0, 0].grid(True, alpha=0.3)

    # Plot 2: Uncertainty ranges for median method
    axes1[0, 1].fill_between(
        range(len(pred_median)),
        uncertainty_median[median_idx, 0],
        uncertainty_median[median_idx, 1],
        alpha=0.3,
        color="steelblue",
        label="Uncertainty Range",
    )
    axes1[0, 1].scatter(
        range(len(pred_median)),
        pred_median[median_idx],
        s=10,
        color="steelblue",
        alpha=0.7,
        label="Prediction",
    )
    axes1[0, 1].set_title("VennAbers-median: Predictions with Uncertainty Ranges")
    axes1[0, 1].set_xlabel("Samples (sorted by prediction)")
    axes1[0, 1].set_ylabel("Probability")
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)

    # Plot 3: Range width histogram for mean method
    range_width_mean = uncertainty_mean[:, 1] - uncertainty_mean[:, 0]
    axes1[1, 0].hist(range_width_mean, bins=30, color="darkblue", alpha=0.7)
    axes1[1, 0].set_title("VennAbers-mean: Uncertainty Range Width Distribution")
    axes1[1, 0].set_xlabel("Range Width")
    axes1[1, 0].set_ylabel("Count")
    axes1[1, 0].axvline(
        np.mean(range_width_mean),
        color="red",
        linestyle="dashed",
        label=f"Mean: {np.mean(range_width_mean):.4f}",
    )
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].legend()

    # Plot 4: Range width histogram for median method
    range_width_median = uncertainty_median[:, 1] - uncertainty_median[:, 0]
    axes1[1, 1].hist(range_width_median, bins=30, color="steelblue", alpha=0.7)
    axes1[1, 1].set_title("VennAbers-median: Uncertainty Range Width Distribution")
    axes1[1, 1].set_xlabel("Range Width")
    axes1[1, 1].set_ylabel("Count")
    axes1[1, 1].axvline(
        np.mean(range_width_median),
        color="red",
        linestyle="dashed",
        label=f"Mean: {np.mean(range_width_median):.4f}",
    )
    axes1[1, 1].grid(True, alpha=0.3)
    axes1[1, 1].legend()

    # Adjust layout and save figure 1
    plt.tight_layout()
    plt.savefig(
        f"my_results/{mimic_size}/{dataset}_venn_abers_uncertainty.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig1)

    # Create second figure for improved Venn-Abers models
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))

    # Sort by prediction for better visualization
    bal_idx = np.argsort(pred_improved_bal)
    temp_idx = np.argsort(pred_improved_temp)

    # Plot 1: Uncertainty ranges for balanced method
    axes2[0, 0].fill_between(
        range(len(pred_improved_bal)),
        uncertainty_improved_bal[bal_idx, 0],
        uncertainty_improved_bal[bal_idx, 1],
        alpha=0.3,
        color="teal",
        label="Uncertainty Range",
    )
    axes2[0, 0].scatter(
        range(len(pred_improved_bal)),
        pred_improved_bal[bal_idx],
        s=10,
        color="teal",
        alpha=0.7,
        label="Prediction",
    )
    axes2[0, 0].set_title("ImprovedVA-bal: Predictions with Uncertainty Ranges")
    axes2[0, 0].set_xlabel("Samples (sorted by prediction)")
    axes2[0, 0].set_ylabel("Probability")
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)

    # Plot 2: Uncertainty ranges for temperature method
    axes2[0, 1].fill_between(
        range(len(pred_improved_temp)),
        uncertainty_improved_temp[temp_idx, 0],
        uncertainty_improved_temp[temp_idx, 1],
        alpha=0.3,
        color="navy",
        label="Uncertainty Range",
    )
    axes2[0, 1].scatter(
        range(len(pred_improved_temp)),
        pred_improved_temp[temp_idx],
        s=10,
        color="navy",
        alpha=0.7,
        label="Prediction",
    )
    axes2[0, 1].set_title("ImprovedVA-temp: Predictions with Uncertainty Ranges")
    axes2[0, 1].set_xlabel("Samples (sorted by prediction)")
    axes2[0, 1].set_ylabel("Probability")
    axes2[0, 1].legend()
    axes2[0, 1].grid(True, alpha=0.3)

    # Plot 3: Range width histogram for balanced method
    range_width_bal = uncertainty_improved_bal[:, 1] - uncertainty_improved_bal[:, 0]
    axes2[1, 0].hist(range_width_bal, bins=30, color="teal", alpha=0.7)
    axes2[1, 0].set_title("ImprovedVA-bal: Uncertainty Range Width Distribution")
    axes2[1, 0].set_xlabel("Range Width")
    axes2[1, 0].set_ylabel("Count")
    axes2[1, 0].axvline(
        np.mean(range_width_bal),
        color="red",
        linestyle="dashed",
        label=f"Mean: {np.mean(range_width_bal):.4f}",
    )
    axes2[1, 0].grid(True, alpha=0.3)
    axes2[1, 0].legend()

    # Plot 4: Range width histogram for temperature method
    range_width_temp = uncertainty_improved_temp[:, 1] - uncertainty_improved_temp[:, 0]
    axes2[1, 1].hist(range_width_temp, bins=30, color="navy", alpha=0.7)
    axes2[1, 1].set_title("ImprovedVA-temp: Uncertainty Range Width Distribution")
    axes2[1, 1].set_xlabel("Range Width")
    axes2[1, 1].set_ylabel("Count")
    axes2[1, 1].axvline(
        np.mean(range_width_temp),
        color="red",
        linestyle="dashed",
        label=f"Mean: {np.mean(range_width_temp):.4f}",
    )
    axes2[1, 1].grid(True, alpha=0.3)
    axes2[1, 1].legend()

    # Adjust layout and save figure 2
    plt.tight_layout()
    plt.savefig(
        f"my_results/{mimic_size}/{dataset}_improved_venn_abers_uncertainty.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Create calibration report for improved models
    if hasattr(improved_va_balanced, "calibration_report"):
        report_bal = improved_va_balanced.calibration_report()
        report_temp = improved_va_temp.calibration_report()

        # Create third figure for calibration curves
        fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))

        # Plot calibration curves
        if "calibration_curve" in report_bal:
            y_true_bal = report_bal["calibration_curve"]["y_true"]
            y_pred_bal = report_bal["calibration_curve"]["y_pred"]

            axes3[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            axes3[0].plot(
                y_pred_bal,
                y_true_bal,
                marker="o",
                markersize=8,
                linestyle="-",
                color="teal",
                linewidth=2,
                label=f'ImprovedVA-bal (brier={report_bal.get("calibration_mse", 0):.4f})',
            )

            axes3[0].set_xlabel("Predicted probability")
            axes3[0].set_ylabel("True probability in each bin")
            axes3[0].set_title("Calibration Curve - Balanced Model")
            axes3[0].legend(loc="best")
            axes3[0].grid(True, alpha=0.3)

        if "calibration_curve" in report_temp:
            y_true_temp = report_temp["calibration_curve"]["y_true"]
            y_pred_temp = report_temp["calibration_curve"]["y_pred"]

            axes3[1].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            axes3[1].plot(
                y_pred_temp,
                y_true_temp,
                marker="o",
                markersize=8,
                linestyle="-",
                color="navy",
                linewidth=2,
                label=f'ImprovedVA-temp (brier={report_temp.get("calibration_mse", 0):.4f})',
            )

            axes3[1].set_xlabel("Predicted probability")
            axes3[1].set_ylabel("True probability in each bin")
            axes3[1].set_title("Calibration Curve - Temperature Model")
            axes3[1].legend(loc="best")
            axes3[1].grid(True, alpha=0.3)

        # Adjust layout and save figure 3
        plt.tight_layout()
        plt.savefig(
            f"my_results/{mimic_size}/{dataset}_improved_venn_abers_calibration.png",
            dpi=300,
            bbox_inches="tight",
        )


def main(dataset):
    # Create my_results directory if it doesn't exist
    os.makedirs("my_results", exist_ok=True)
    mimic_size = "1000_subjects"
    os.makedirs(f"my_results/{mimic_size}", exist_ok=True)
    EXPERIMENTS = 1

    # timer
    start_time = time.time()
    # For a quick test, just run a single experiment
    results = [
        run_experiment(dataset, rs, mimic_size=mimic_size) for rs in range(EXPERIMENTS)
    ]
    results_df = pd.concat(results, ignore_index=True)

    end_time = time.time()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] Time taken for {EXPERIMENTS} runs of {dataset} with {mimic_size}: {end_time - start_time:.2f} seconds"
    )

    # Save single result
    results_df.to_csv(
        f"my_results/{mimic_size}/{dataset}_improved_model_quick_test.csv", index=False
    )

    # Plot results
    plot_results(results_df, dataset, mimic_size)

    # Plot Venn-Abers uncertainty ranges
    X_train, X_test, y_train, y_test = read_data(
        dataset, random_seed=0, mimic_size=mimic_size
    )
    plot_uncertainty_ranges(dataset, X_test, y_test, mimic_size)

    # Print summary comparison
    # print("\nSUMMARY COMPARISON:")
    # print(
    #     results_df[
    #         [
    #             "model",
    #             "brier_scaled",
    #             "hosmer_lemshow",
    #             "reliability_small",
    #             "reliability_large",
    #         ]
    #     ]
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    main(args.dataset)
