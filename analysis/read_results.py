import pandas as pd
import argparse
import os


def main(dataset):
    # Find available results
    result_files = []
    for root, dirs, files in os.walk("analysis/my_results"):
        for file in files:
            if file.endswith('_improved_model_quick_test.csv'):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print("No result files found in analysis/my_results directory.")
        return
        
    # If dataset is not specified or not found, use the first available file
    if dataset is None:
        result_file = result_files[0]
        print(f"Using result file: {result_file}")
    else:
        matching_files = [f for f in result_files if dataset in f]
        if not matching_files:
            print(f"No results found for dataset '{dataset}'. Using first available file.")
            result_file = result_files[0]
        else:
            result_file = matching_files[0]
        
    # Read the CSV file
    df = pd.read_csv(result_file)

    # Print all rows to see full comparison
    print(df)

    # Compare models based on key metrics
    print("\nModel Comparison:")
    # For AUC, higher is better
    print("\nBest model by AUC (higher is better):")
    best_auc_idx = df["auc"].idxmax()
    print(
        f"{df.iloc[best_auc_idx]['model']} with AUC = {df.iloc[best_auc_idx]['auc']:.6f}"
    )

    # Ignore extreme values for brier_scaled (like -8.9) that indicate major issues
    valid_scaled_df = df[df["brier_scaled"] > -1]
    best_scaled_idx = valid_scaled_df["brier_scaled"].idxmax()
    print(
        f"\nBest model by brier_scaled (higher is better, excluding negative values):"
    )
    print(
        f"{df.iloc[best_scaled_idx]['model']} with brier_scaled = {df.iloc[best_scaled_idx]['brier_scaled']:.6f}"
    )

    # For other metrics, lower is better
    for metric in ["brier", "reliability_small", "reliability_large"]:
        print(f"\nBest model by {metric} (lower is better):")
        best_idx = df[metric].idxmin()
        print(
            f"{df.iloc[best_idx]['model']} with {metric} = {df.iloc[best_idx][metric]:.6f}"
        )

    # hosmer_lemshow p_value
    metric = "hosmer_lemshow"
    print(f"\nBest model by {metric} (higher is better & >.05):")
    best_idx = df[metric].idxmax()
    print(
        f"{df.iloc[best_idx]['model']} with {metric} = {df.iloc[best_idx][metric]:.6f}"
    )

    # Print detailed comparison of ImprovedCF models vs others
    print("\nDetailed Model Comparison:")
    print(
        "\n                 AUC    Brier  Brier_scaled  Reliability_small  Reliability_large  Hosmer-Lemeshow  Spiegelhalter"
    )
    for idx, row in df.iterrows():
        model = row["model"]
        auc = row["auc"]
        brier = row["brier"]
        brier_scaled = row["brier_scaled"]
        rel_small = row["reliability_small"]
        rel_large = row["reliability_large"]
        hosmer = row["hosmer_lemshow"]
        spiegel = row["speigelhalter"]

        print(
            f"{model:15} {auc:.4f}  {brier:.6f}  {brier_scaled:11.6f}  {rel_small:16.6f}  {rel_large:17.6f}  {hosmer:15.6f}  {spiegel:.6f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, 
                        help="Dataset name to filter results. If not provided, will use first available file.")
    args = parser.parse_args()
    main(args.dataset)
