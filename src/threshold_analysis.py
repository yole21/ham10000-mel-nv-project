import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0
    balanced_acc = (sensitivity + specificity) / 2.0
    youden_j = sensitivity + specificity - 1.0

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "youden_j": youden_j,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True, help="Path to prediction CSV")
    parser.add_argument("--out_dir", type=str, default="outputs/thresholds", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.pred_csv)

    y_true = df["true_label"].values
    y_prob = df["prob_mel"].values

    thresholds = np.linspace(0.0, 1.0, 101)

    rows = []
    for th in thresholds:
        metrics = compute_metrics(y_true, y_prob, th)
        rows.append(metrics)

    result_df = pd.DataFrame(rows)

    base_name = os.path.splitext(os.path.basename(args.pred_csv))[0]
    csv_path = os.path.join(args.out_dir, f"{base_name}_threshold_metrics.csv")
    result_df.to_csv(csv_path, index=False)

    # Best thresholds
    best_youden = result_df.loc[result_df["youden_j"].idxmax()]
    best_bal_acc = result_df.loc[result_df["balanced_accuracy"].idxmax()]
    best_f1 = result_df.loc[result_df["f1"].idxmax()]

    print("\n=== Best Thresholds ===")
    print(f"Youden's J best threshold       : {best_youden['threshold']:.2f}")
    print(f"Sensitivity / Specificity      : {best_youden['sensitivity']:.4f} / {best_youden['specificity']:.4f}")
    print(f"Balanced Accuracy / F1         : {best_youden['balanced_accuracy']:.4f} / {best_youden['f1']:.4f}")

    print()
    print(f"Balanced Accuracy best threshold: {best_bal_acc['threshold']:.2f}")
    print(f"Sensitivity / Specificity      : {best_bal_acc['sensitivity']:.4f} / {best_bal_acc['specificity']:.4f}")
    print(f"Balanced Accuracy / F1         : {best_bal_acc['balanced_accuracy']:.4f} / {best_bal_acc['f1']:.4f}")

    print()
    print(f"F1 best threshold              : {best_f1['threshold']:.2f}")
    print(f"Sensitivity / Specificity      : {best_f1['sensitivity']:.4f} / {best_f1['specificity']:.4f}")
    print(f"Balanced Accuracy / F1         : {best_f1['balanced_accuracy']:.4f} / {best_f1['f1']:.4f}")

    # Threshold closest to sensitivity 0.80
    target_sens = 0.80
    idx_target = (result_df["sensitivity"] - target_sens).abs().idxmin()
    target_row = result_df.loc[idx_target]

    print()
    print(f"Threshold closest to sensitivity {target_sens:.2f}: {target_row['threshold']:.2f}")
    print(f"Sensitivity / Specificity      : {target_row['sensitivity']:.4f} / {target_row['specificity']:.4f}")
    print(f"Balanced Accuracy / F1         : {target_row['balanced_accuracy']:.4f} / {target_row['f1']:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(result_df["threshold"], result_df["sensitivity"], label="Sensitivity")
    plt.plot(result_df["threshold"], result_df["specificity"], label="Specificity")
    plt.plot(result_df["threshold"], result_df["precision"], label="Precision")
    plt.plot(result_df["threshold"], result_df["f1"], label="F1-score")
    plt.plot(result_df["threshold"], result_df["balanced_accuracy"], label="Balanced Accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title(f"Threshold Analysis: {base_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, f"{base_name}_threshold_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nSaved threshold metrics to: {csv_path}")
    print(f"Saved threshold plot to   : {plot_path}")


if __name__ == "__main__":
    main()