import numpy as np
import pandas as pd
import argparse
import time
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Import all model variants
from califorest import (
    CaliForest,
    RC30,
    VennAbersForest,
    ImprovedVennAbersForest,
    BayesianVennAbersForest,
    STLBRF,
)
from califorest.stlbrf2 import STLBRF2
from califorest.stlbrf_oob import STLBRF_OOB
from califorest.rfva import RFVA
from califorest import metrics as em
from run_chil_exp import read_data

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# For saving results
import datetime
import json


def init_all_models(n_estimators=300, max_depth=10, random_state=42):
    """
    Initialize all model variants with consistent hyperparameters.

    Parameters:
    -----------
    n_estimators : int, default=300
        Number of trees in each forest-based model
    max_depth : int, default=10
        Maximum depth of trees
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary mapping model names to initialized model objects
    """
    min_samples_split = 3
    min_samples_leaf = 1

    models = {
        "RF": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ),
        "CaliForest-Iso": CaliForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ctype="isotonic",
        ),
        "CaliForest-Logit": CaliForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            ctype="logistic",
        ),
        "RFVA": RFVA(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        ),
        # "STLBRF": STLBRF(
        #     n_estimators=n_estimators,
        #     max_depth=max_depth,
        #     min_samples_split=min_samples_split,
        #     min_samples_leaf=min_samples_leaf,
        #     random_state=random_state,
        # ),
        "STLBRF2": STLBRF2(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ),
        "STLBRF_OOB": STLBRF_OOB(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        ),
        "VennAbers": VennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        ),
        "ImprovedVennAbers": ImprovedVennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        ),
        "BayesianVennAbers": BayesianVennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        ),
        "VennAbersForest": VennAbersForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        ),
    }

    return models


def evaluate_models(
    dataset,
    model_names=None,
    n_seeds=5,
    results_dir="results",
    mimic_size="1000_subjects",
):
    """
    Evaluate selected models on a dataset with multiple seeds.

    Parameters:
    -----------
    dataset : str
        Dataset name to evaluate on
    model_names : list of str, default=None
        Names of models to evaluate. If None, evaluates all models.
    n_seeds : int, default=5
        Number of random seeds to use
    results_dir : str, default="results"
        Directory to save results
    mimic_size : str, default="5000_subjects"
        Size for MIMIC datasets

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing evaluation results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Results to collect
    results = []
    headers = [
        "dataset",
        "model",
        "random_seed",
        "train_time",
        "inference_time",
        "auc",
        "brier",
        "brier_scaled",
        "hosmer_lemeshow",
        "spiegelhalter",
        "reliability_small",
        "reliability_large",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    print(f"Evaluating models on {dataset} dataset")

    # Run with different random seeds
    for seed in range(n_seeds):
        print(f"\nSeed {seed+1}/{n_seeds}")

        # Read data
        print(f"  Loading data...")
        X_train, X_test, y_train, y_test = read_data(
            dataset, seed, mimic_size=mimic_size
        )
        print(
            f"  Data loaded. Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        # Initialize models
        all_models = init_all_models(random_state=seed)

        # Use specified models or all models
        if model_names:
            models_to_evaluate = {
                name: all_models[name] for name in model_names if name in all_models
            }
            if len(models_to_evaluate) < len(model_names):
                missing = set(model_names) - set(models_to_evaluate.keys())
                print(f"  Warning: Models not found: {missing}")
        else:
            models_to_evaluate = all_models

        # Evaluate each model
        for name, model in models_to_evaluate.items():
            print(f"  Evaluating {name}...")

            try:
                # Train model
                t_start = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t_start

                # Test model
                t_start = time.time()
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                inference_time = time.time() - t_start

                # Compute metrics
                score_auc = roc_auc_score(y_test, y_prob)
                score_hl = em.hosmer_lemeshow(y_test, y_prob)
                score_sh = em.spiegelhalter(y_test, y_prob)
                score_b, score_bs = em.scaled_brier_score(y_test, y_prob)
                rel_small, rel_large = em.reliability(y_test, y_prob)

                # simple metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Save result
                row = [
                    dataset,
                    name,
                    seed,
                    train_time,
                    inference_time,
                    score_auc,
                    score_b,
                    score_bs,
                    score_hl,
                    score_sh,
                    rel_small,
                    rel_large,
                    accuracy,
                    precision,
                    recall,
                    f1,
                ]
                results.append(row)

                print(
                    f"    AUC: {score_auc:.4f}, Brier: {score_b:.4f}, Time: {train_time:.2f}s"
                )

            except Exception as e:
                print(f"    Error evaluating {name}: {str(e)}")

    # Create DataFrame and save results
    df = pd.DataFrame(results, columns=headers)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"{dataset}_results_{timestamp}.csv")
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    return df


def _create_chart(agg_results, title, metric, dataset, timestamp, output_dir):
    """
    Plot a chart for a given metric.
    """
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        agg_results["model"],
        agg_results[f"{metric}_mean"],
        yerr=agg_results[f"{metric}_std"],
    )
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")

    # Add text labels above each bar
    for bar in bars:
        yval = bar.get_height()
        # Place text slightly above the bar, centered horizontally
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.3f}",
            va="bottom",
            ha="center",
        )

    # Adjust y-limit to make space for labels if needed
    plt.ylim(top=plt.ylim()[1] * 1.05)  # Increase upper y-limit by 5%

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_{metric}_{timestamp}.png"))
    plt.close()  # Close the plot after saving


def visualize_results(results_df, output_dir="figures"):
    """
    Create visualizations from results.

    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame containing evaluation results
    output_dir : str, default="figures"
        Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Dataset name
    dataset = results_df["dataset"].iloc[0]

    # Compute aggregated metrics
    agg_results = results_df.groupby("model").agg(
        {
            "auc": ["mean", "std"],
            "brier": ["mean", "std"],
            "brier_scaled": ["mean", "std"],
            "hosmer_lemeshow": ["mean", "std"],
            "spiegelhalter": ["mean", "std"],
            "reliability_small": ["mean", "std"],
            "reliability_large": ["mean", "std"],
            "train_time": ["mean", "std"],
            "inference_time": ["mean", "std"],
            "accuracy": ["mean", "std"],
            "precision": ["mean", "std"],
            "recall": ["mean", "std"],
            "f1": ["mean", "std"],
        }
    )

    # Reset index for easier plotting
    agg_results.columns = ["_".join(col).strip() for col in agg_results.columns.values]
    agg_results = agg_results.reset_index()

    # Sort by AUC (descending)
    agg_results = agg_results.sort_values("auc_mean", ascending=False)

    plot_map = {
        "auc": f"{dataset} - Mean AUC Scores (higher is better)",
        "brier": f"{dataset} - Mean Brier Scores (lower is better)",
        "brier_scaled": f"{dataset} - Mean Scaled Brier Scores (lower is better)",
        "hosmer_lemeshow": f"{dataset} - Mean Hosmer-Lemeshow p-values (higher is better)",
        "spiegelhalter": f"{dataset} - Mean Spiegelhalter p-values (higher is better)",
        "reliability_small": f"{dataset} - Mean Reliability-in-the-small (lower is better)",
        "reliability_large": f"{dataset} - Mean Reliability-in-the-large (higher is better)",
        "accuracy": f"{dataset} - Mean Accuracy (higher is better)",
        "precision": f"{dataset} - Mean Precision (higher is better)",
        "recall": f"{dataset} - Mean Recall (higher is better)",
        "f1": f"{dataset} - Mean F1 (higher is better)",
        "train_time": f"{dataset} - Mean Training Times",
    }

    for metric, title in plot_map.items():
        _create_chart(
            agg_results,
            title,
            metric,
            dataset,
            timestamp,
            output_dir,
        )

    # Save aggregated results
    agg_results.to_csv(
        os.path.join(output_dir, f"{dataset}_agg_results_{timestamp}.csv"), index=False
    )
    print(f"Visualizations saved to {output_dir}")


def plot_calibration_curves(
    dataset, model_names=None, seed=42, output_dir="figures", mimic_size="5000_subjects"
):
    """
    Plot calibration curves for models.

    Parameters:
    -----------
    dataset : str
        Dataset name
    model_names : list of str, default=None
        Names of models to evaluate. If None, evaluates all models.
    seed : int, default=42
        Random seed
    output_dir : str, default="figures"
        Directory to save figures
    mimic_size : str, default="5000_subjects"
        Size for MIMIC datasets
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Read data
    X_train, X_test, y_train, y_test = read_data(dataset, seed, mimic_size=mimic_size)

    # Initialize models
    all_models = init_all_models(random_state=seed)

    # Use specified models or all models
    if model_names:
        models_to_evaluate = {
            name: all_models[name] for name in model_names if name in all_models
        }
    else:
        models_to_evaluate = all_models

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Evaluate each model
    for name, model in models_to_evaluate.items():
        print(f"Calibration curve for {name}...")

        try:
            # Train model
            model.fit(X_train, y_train)

            # Get predictions
            y_pred = model.predict_proba(X_test)[:, 1]

            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(y_test, y_pred, n_bins=10)

            # Plot calibration curve
            ax1.plot(prob_pred, prob_true, "s-", label=f"{name}")

            # Plot histogram
            ax2.hist(y_pred, range=(0, 1), bins=10, histtype="step", lw=2, label=name)

        except Exception as e:
            print(f"  Error with {name}: {str(e)}")

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f"{dataset} - Calibration curves")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{dataset}_calibration_curves_{timestamp}.png")
    )
    print(f"Calibration curves saved to {output_dir}")


def main():
    """Main function to run the model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare different random forest models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mimic3_mort_icu",
        help="Dataset to use (e.g., breast_cancer, iris, hastie, mimic3_mort_icu)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of random seeds to use"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate (if not specified, all models will be evaluated)",
    )
    parser.add_argument(
        "--mimic_size",
        type=str,
        default="1000_subjects",
        help="Size for MIMIC dataset (e.g., 10000_subjects, 5000_subjects)",
    )
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--figures_dir", type=str, default="figures", help="Directory to save figures"
    )
    parser.add_argument(
        "--calibration_curves", action="store_true", help="Generate calibration curves"
    )

    args = parser.parse_args()

    # Log arguments
    print("Running with arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Evaluate models
    results_df = evaluate_models(
        dataset=args.dataset,
        model_names=args.models,
        n_seeds=args.seeds,
        results_dir=args.results_dir,
        mimic_size=args.mimic_size,
    )

    output_dir = f"{args.figures_dir}/{args.dataset}/{args.mimic_size}"

    # Visualize results
    visualize_results(results_df, output_dir=output_dir)

    # Plot calibration curves if requested
    if args.calibration_curves:
        plot_calibration_curves(
            dataset=args.dataset,
            model_names=args.models,
            seed=0,  # Use first seed
            output_dir=output_dir,
            mimic_size=args.mimic_size,
        )

    print("Model comparison completed successfully!")


if __name__ == "__main__":
    main()
