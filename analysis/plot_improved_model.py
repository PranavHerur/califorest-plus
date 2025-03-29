#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main(dataset="hastie"):
    # Read the data
    data = pd.read_csv(f"my_results/{dataset}.csv")

    # Set style for plots
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))

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
        "CF-Logit-Improved": "lightblue",
        "RC-Iso": "indianred",
        "RC-Logit": "sandybrown",
        "RF-NoCal": "lightgray",
    }

    # Plot each metric
    for metric_name, title, ax in metrics:
        # Create boxplot
        sns.boxplot(
            x="model",
            y=metric_name,
            hue="model",
            data=data,
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
    plt.savefig(f"{dataset}-results-reproduced.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()
    main(args.dataset)
