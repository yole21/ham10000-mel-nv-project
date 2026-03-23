import os
import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (sensitivity + specificity) / 2.0
    f1 = f1_score(y_true, y_pred, zero_division=0)

    auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return {
        "accuracy": acc,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "auc": auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", type=str, required=True, help="Prediction CSV path")
    parser.add_argument("--meta_csv", type=str, default="data/raw/HAM10000_metadata.csv", help="Metadata CSV path")
    parser.add_argument("--agg", type=str, default="mean", choices=["mean", "max"], help="Lesion aggregation method")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--out_dir", type=str, default="outputs/lesion_level", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pred_df = pd.read_csv(args.pred_csv)
    meta_df = pd.read_csv(args.meta_csv)

    merged = pred_df.merge(meta_df[["image_id", "lesion_id"]], on="image_id", how="left")

    if merged["lesion_id"].isna().any():
        missing_count = merged["lesion_id"].isna().sum()
        raise ValueError(f"{missing_count} rows could not be matched with lesion_id.")

    lesion_true = merged.groupby("lesion_id")["true_label"].first()

    if args.agg == "mean":
        lesion_prob = merged.groupby("lesion_id")["prob_mel"].mean()
    else:
        lesion_prob = merged.groupby("lesion_id")["prob_mel"].max()

    lesion_df = pd.DataFrame({
        "lesion_id": lesion_true.index,
        "true_label": lesion_true.values,
        "prob_mel": lesion_prob.values,
    })

    metrics = compute_metrics(
        lesion_df["true_label"].values,
        lesion_df["prob_mel"].values,
        threshold=args.threshold
    )

    base_name = os.path.splitext(os.path.basename(args.pred_csv))[0]

    lesion_csv_path = os.path.join(args.out_dir, f"{base_name}_lesion_{args.agg}_predictions.csv")
    lesion_df.to_csv(lesion_csv_path, index=False)

    metrics_path = os.path.join(args.out_dir, f"{base_name}_lesion_{args.agg}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Aggregation: {args.agg}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print("\n=== Lesion-level Results ===")
    print(f"Aggregation method : {args.agg}")
    print(f"Threshold          : {args.threshold}")
    print(f"Number of lesions  : {len(lesion_df)}")
    print()
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Sensitivity      : {metrics['sensitivity']:.4f}")
    print(f"Specificity      : {metrics['specificity']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1-score         : {metrics['f1']:.4f}")
    print(f"AUC              : {metrics['auc']:.4f}")
    print(f"PR-AUC           : {metrics['pr_auc']:.4f}")
    print()
    print("=== Confusion Matrix ===")
    print(f"TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TP: {metrics['tp']}")
    print()
    print(f"Saved lesion-level predictions to: {lesion_csv_path}")
    print(f"Saved lesion-level metrics to    : {metrics_path}")


if __name__ == "__main__":
    main()