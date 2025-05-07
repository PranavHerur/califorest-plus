import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

output_dir = "figures/mimic3_mort_hosp/full_mimic3/"
timestamp = "20250505_105250"
target = "mimic3_mort_hosp"
comparison_df = pd.read_csv(f"{output_dir}/{target}_agg_results_{timestamp}.csv")

# Plot comparison metrics
plt.figure(figsize=(18, 10))

# Define metrics to plot
metrics = [
    ("auc_mean", "AUC (higher is better)"),
    ("brier_mean", "Brier Score (lower is better)"),
    ("brier_scaled_mean", "Scaled Brier Score (higher is better)"),
    ("f1_mean", "F1 Score (higher is better)"),
]

for i, (metric, title) in enumerate(metrics):
    plt.subplot(2, 2, i + 1)

    # For metrics where higher is better
    if metric in ["auc_mean", "brier_scaled_mean"]:
        # Sort by metric value descending
        # sorted_df = comparison_df.sort_values(by=metric, ascending=False)
        colors = sns.color_palette("YlGn", len(comparison_df))
    else:
        # Sort by metric value ascending (lower is better)
        # sorted_df = comparison_df.sort_values(by=metric)
        colors = sns.color_palette("YlOrRd_r", len(comparison_df))

    # Create bar plot
    ax = sns.barplot(x="model", y=metric, data=comparison_df, palette=colors)

    # Add value labels on bars
    for j, p in enumerate(ax.patches):
        format_str = ".4f" if metric != "training_time" else ".2f"
        ax.annotate(
            f"{p.get_height():{format_str}}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=45,
        )

    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

plt.savefig(
    f"{output_dir}/{target}_metrics_comparison_{timestamp}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
