#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from run_chil_exp import read_data
from califorest import CaliForest, RC30, ImprovedCaliForest
from califorest import metrics as em

# Define constants
n_estimators = 300
max_depth = 5
min_samples_split = 3
min_samples_leaf = 1
random_seed = 42


def run_experiment(dataset, random_seed=42):
    """Run experiment on the dataset with different models"""

    X_train, X_test, y_train, y_test = read_data(dataset, random_seed=random_seed)

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
        "ImprovedCF": ImprovedCaliForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ctype="isotonic",
            ensemble_weight=0.7,  # Adjust weight to improve reliability
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
        print(f"Training {name}...")
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
        print(f"  AUC: {score_auc:.4f}")
        print(f"  Brier Score: {score_b:.4f}")
        print(f"  Scaled Brier Score: {score_bs:.4f}")
        print(f"  Hosmer-Lemeshow p-value: {score_hl:.4f}")
        print(f"  Spiegelhalter p-value: {score_sh:.4f}")
        print(f"  Reliability-in-the-small: {rel_small:.6f}")
        print(f"  Reliability-in-the-large: {rel_large:.6f}")
        print()

    return pd.DataFrame(results)


def run_multiple_seeds(n_seeds=5):
    """Run experiment with multiple random seeds"""
    all_results = []

    for seed in range(n_seeds):
        print(f"Running with seed {seed}...")
        seed_results = run_experiment(dataset, random_seed=seed)
        all_results.append(seed_results)

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save results
    results_df.to_csv("my_results/improved_model_results.csv", index=False)

    return results_df


def plot_results(results_df):
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
        "ImprovedCF": "purple",
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

        # Remove x-axis label
        ax.set_xlabel("model")

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("my_results/improved_model_results.png", dpi=300, bbox_inches="tight")
    plt.show()


def main(dataset):
    # Create my_results directory if it doesn't exist
    os.makedirs("my_results", exist_ok=True)

    # For a quick test, just run a single experiment
    results_df = run_experiment(dataset, random_seed=42)

    # Save single result
    results_df.to_csv("my_results/improved_model_quick_test.csv", index=False)

    # Print summary comparison
    print("\nSUMMARY COMPARISON:")
    print(
        results_df[
            [
                "model",
                "brier_scaled",
                "hosmer_lemshow",
                "reliability_small",
                "reliability_large",
            ]
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    main(args.dataset)
