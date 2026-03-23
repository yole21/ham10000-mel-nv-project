import os
import re
import glob
import pandas as pd


def extract_model_and_seed(filename):
    """
    Example:
    freeze_ce_seed42_test_predictions_threshold_metrics.csv
    -> model_name='freeze_ce', seed=42
    """
    base = os.path.basename(filename)
    pattern = r"^(.*?)_seed(\d+)_test_predictions_threshold_metrics\.csv$"
    match = re.match(pattern, base)
    if not match:
        return None, None
    model_name = match.group(1)
    seed = int(match.group(2))
    return model_name, seed


def summarize_one_file(csv_path, target_sens=0.80):
    df = pd.read_csv(csv_path)

    best_youden = df.loc[df["youden_j"].idxmax()]
    best_bal_acc = df.loc[df["balanced_accuracy"].idxmax()]
    best_f1 = df.loc[df["f1"].idxmax()]

    idx_target = (df["sensitivity"] - target_sens).abs().idxmin()
    target_row = df.loc[idx_target]

    return {
        "best_youden_threshold": best_youden["threshold"],
        "best_youden_sensitivity": best_youden["sensitivity"],
        "best_youden_specificity": best_youden["specificity"],
        "best_youden_bal_acc": best_youden["balanced_accuracy"],
        "best_youden_f1": best_youden["f1"],

        "best_bal_acc_threshold": best_bal_acc["threshold"],
        "best_bal_acc_sensitivity": best_bal_acc["sensitivity"],
        "best_bal_acc_specificity": best_bal_acc["specificity"],
        "best_bal_acc_value": best_bal_acc["balanced_accuracy"],
        "best_bal_acc_f1": best_bal_acc["f1"],

        "best_f1_threshold": best_f1["threshold"],
        "best_f1_sensitivity": best_f1["sensitivity"],
        "best_f1_specificity": best_f1["specificity"],
        "best_f1_bal_acc": best_f1["balanced_accuracy"],
        "best_f1_value": best_f1["f1"],

        "target_sens_threshold": target_row["threshold"],
        "target_sens_value": target_row["sensitivity"],
        "target_sens_specificity": target_row["specificity"],
        "target_sens_bal_acc": target_row["balanced_accuracy"],
        "target_sens_f1": target_row["f1"],
    }


def format_mean_std(series):
    return f"{series.mean():.4f} ± {series.std(ddof=1):.4f}"


def main():
    threshold_dir = "outputs/thresholds"
    out_dir = "outputs/threshold_summary"
    os.makedirs(out_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(threshold_dir, "*_threshold_metrics.csv")))

    if not csv_files:
        print("No threshold metric CSV files found.")
        return

    run_rows = []
    for csv_path in csv_files:
        model_name, seed = extract_model_and_seed(csv_path)
        if model_name is None:
            print(f"Skipping unmatched file: {csv_path}")
            continue

        summary = summarize_one_file(csv_path, target_sens=0.80)
        row = {
            "model": model_name,
            "seed": seed,
            "file": os.path.basename(csv_path),
            **summary,
        }
        run_rows.append(row)

    run_df = pd.DataFrame(run_rows).sort_values(["model", "seed"])
    run_csv_path = os.path.join(out_dir, "threshold_run_level_summary.csv")
    run_df.to_csv(run_csv_path, index=False)

    summary_rows = []
    for model_name, g in run_df.groupby("model"):
        summary_rows.append({
            "model": model_name,

            "best_youden_threshold": format_mean_std(g["best_youden_threshold"]),
            "best_youden_sensitivity": format_mean_std(g["best_youden_sensitivity"]),
            "best_youden_specificity": format_mean_std(g["best_youden_specificity"]),
            "best_youden_bal_acc": format_mean_std(g["best_youden_bal_acc"]),
            "best_youden_f1": format_mean_std(g["best_youden_f1"]),

            "best_bal_acc_threshold": format_mean_std(g["best_bal_acc_threshold"]),
            "best_bal_acc_sensitivity": format_mean_std(g["best_bal_acc_sensitivity"]),
            "best_bal_acc_specificity": format_mean_std(g["best_bal_acc_specificity"]),
            "best_bal_acc_value": format_mean_std(g["best_bal_acc_value"]),
            "best_bal_acc_f1": format_mean_std(g["best_bal_acc_f1"]),

            "best_f1_threshold": format_mean_std(g["best_f1_threshold"]),
            "best_f1_sensitivity": format_mean_std(g["best_f1_sensitivity"]),
            "best_f1_specificity": format_mean_std(g["best_f1_specificity"]),
            "best_f1_bal_acc": format_mean_std(g["best_f1_bal_acc"]),
            "best_f1_value": format_mean_std(g["best_f1_value"]),

            "target_sens_threshold": format_mean_std(g["target_sens_threshold"]),
            "target_sens_value": format_mean_std(g["target_sens_value"]),
            "target_sens_specificity": format_mean_std(g["target_sens_specificity"]),
            "target_sens_bal_acc": format_mean_std(g["target_sens_bal_acc"]),
            "target_sens_f1": format_mean_std(g["target_sens_f1"]),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("model")
    summary_csv_path = os.path.join(out_dir, "threshold_model_mean_std_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nSaved run-level summary to:")
    print(run_csv_path)

    print("\nSaved model mean ± std summary to:")
    print(summary_csv_path)

    print("\n=== Model Mean ± Std Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()