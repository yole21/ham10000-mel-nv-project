import os
import re
import glob
import pandas as pd


def extract_model_and_seed(filename):
    """
    Example:
    finetune_ce_seed42_test_predictions_lesion_mean_metrics.txt
    -> model='finetune_ce', seed=42
    """
    base = os.path.basename(filename)
    pattern = r"^(.*?)_seed(\d+)_test_predictions_lesion_mean_metrics\.txt$"
    match = re.match(pattern, base)
    if not match:
        return None, None
    model = match.group(1)
    seed = int(match.group(2))
    return model, seed


def parse_metrics_txt(txt_path):
    metrics = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key in ["Aggregation", "Threshold"]:
                continue

            try:
                metrics[key] = float(value)
            except ValueError:
                pass

    return metrics


def format_mean_std(series):
    return f"{series.mean():.4f} ± {series.std(ddof=1):.4f}"


def main():
    lesion_dir = "outputs/lesion_level"
    out_dir = "outputs/lesion_summary"
    os.makedirs(out_dir, exist_ok=True)

    txt_files = sorted(glob.glob(os.path.join(lesion_dir, "*_lesion_mean_metrics.txt")))

    if not txt_files:
        print("No lesion-level metric txt files found.")
        return

    rows = []
    for txt_path in txt_files:
        model, seed = extract_model_and_seed(txt_path)
        if model is None:
            print(f"Skipping unmatched file: {txt_path}")
            continue

        metrics = parse_metrics_txt(txt_path)
        row = {
            "model": model,
            "seed": seed,
            "file": os.path.basename(txt_path),
            **metrics
        }
        rows.append(row)

    run_df = pd.DataFrame(rows).sort_values(["model", "seed"])
    run_csv_path = os.path.join(out_dir, "lesion_run_level_summary.csv")
    run_df.to_csv(run_csv_path, index=False)

    metric_cols = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "f1",
        "auc",
        "pr_auc",
    ]

    summary_rows = []
    for model, g in run_df.groupby("model"):
        summary = {"model": model}
        for col in metric_cols:
            summary[col] = format_mean_std(g[col])
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values("model")
    summary_csv_path = os.path.join(out_dir, "lesion_model_mean_std_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nSaved run-level summary to:")
    print(run_csv_path)

    print("\nSaved model mean ± std summary to:")
    print(summary_csv_path)

    print("\n=== Lesion-level Model Mean ± Std Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
    