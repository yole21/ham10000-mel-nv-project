from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from dataset import create_dataloaders
from model import create_resnet18_model


# evaluate_experiment.py
# Evaluate saved checkpoints from train_experiment.py


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained experiment model on test set")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pth model checkpoint")
    parser.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--test_csv", type=str, default="data/splits/test.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="outputs")

    return parser.parse_args()


def infer_freeze_mode_from_filename(model_path: Path) -> bool:
    name = model_path.stem.lower()
    if "freeze" in name:
        return True
    if "finetune" in name:
        return False
    raise ValueError(
        "Could not infer model mode from filename. "
        "Please keep names like freeze_ce_seed42.pth or finetune_weighted_seed42.pth"
    )


def make_output_dirs(base_output_dir: str):
    base = Path(base_output_dir)
    paths = {
        "base": base,
        "metrics": base / "metrics",
        "predictions": base / "predictions",
        "figures": base / "figures",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_model_for_evaluation(model_path: Path, device: torch.device):
    freeze_backbone = infer_freeze_mode_from_filename(model_path)
    model = create_resnet18_model(num_classes=2, freeze_backbone=freeze_backbone)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        checkpoint_info = checkpoint
    else:
        model.load_state_dict(checkpoint)
        checkpoint_info = None

    model = model.to(device)
    return model, checkpoint_info, freeze_backbone


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    all_image_paths = []

    dataset_df = getattr(loader.dataset, "df", None)
    row_start = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        if dataset_df is not None and "image_path" in dataset_df.columns:
            batch_paths = dataset_df.iloc[row_start:row_start + batch_size]["image_path"].tolist()
            all_image_paths.extend(batch_paths)
        else:
            all_image_paths.extend([""] * batch_size)

        row_start += batch_size

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return all_labels, all_preds, all_probs, all_image_paths


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = ((y_true == 1) & (y_pred == 1)).sum() / max((y_true == 1).sum(), 1)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def save_predictions_csv(output_path: Path, y_true, y_pred, y_prob, image_paths):
    rows = []
    for path, true_label, pred_label, prob_mel in zip(image_paths, y_true, y_pred, y_prob):
        path_obj = Path(path) if path else None
        image_id = path_obj.stem if path_obj else ""
        rows.append({
            "image_path": path,
            "image_id": image_id,
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "prob_mel": float(prob_mel),
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_path, index=False)
    return pred_df

def prettify_experiment_name(exp_name: str) -> str:
    pretty = exp_name

    pretty = pretty.replace("freeze_ce", "Freeze + CE")
    pretty = pretty.replace("freeze_weighted", "Freeze + Weighted")
    pretty = pretty.replace("finetune_ce", "Fine-tune + CE")
    pretty = pretty.replace("finetune_weighted", "Fine-tune + Weighted")

    pretty = pretty.replace("_seed42", "")
    pretty = pretty.replace("_seed52", "")
    pretty = pretty.replace("_seed62", "")

    return pretty

def save_confusion_matrix_figure(cm, output_path: Path, title: str):
    plt.figure(figsize=(5.2, 4.6))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=15, pad=10)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    class_names = ["NV", "MEL"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if cm[i, j] > threshold else "black"
            )

    # Add thin borders between cells
    ax.set_xticks(np.arange(-0.5, cm.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cm.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_curve_figure(y_true, y_prob, auc_value: float, output_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(5.2, 4.6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=15, pad=10)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_dirs = make_output_dirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    _, _, test_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model, checkpoint_info, freeze_backbone = load_model_for_evaluation(model_path, device)
    print(f"Loaded model from: {model_path}")
    print(f"Mode inferred from filename: {'freeze' if freeze_backbone else 'finetune'}")

    y_true, _, y_prob, image_paths = evaluate_model(model, test_loader, device)
    y_pred = (y_prob >= args.threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    exp_name = model_path.stem

    predictions_path = output_dirs["predictions"] / f"{exp_name}_test_predictions.csv"
    save_predictions_csv(predictions_path, y_true, y_pred, y_prob, image_paths)

    metrics_output = {
        "experiment_name": exp_name,
        "model_path": str(model_path),
        "threshold": args.threshold,
        "test_csv": args.test_csv,
        "metrics": metrics,
        "predictions_csv": str(predictions_path),
    }

    if checkpoint_info is not None:
        metrics_output["checkpoint_epoch"] = checkpoint_info.get("epoch")
        metrics_output["checkpoint_save_metric"] = checkpoint_info.get("save_metric")
        metrics_output["checkpoint_best_score"] = checkpoint_info.get("best_score")
        metrics_output["train_config"] = checkpoint_info.get("config")

    metrics_json_path = output_dirs["metrics"] / f"{exp_name}_test.json"
    save_json(metrics_output, metrics_json_path)

    # Save figures
    figures_dir = output_dirs["figures"]
    cm_array = np.array(metrics["confusion_matrix"])

    cm_fig_path = figures_dir / f"cm_{exp_name}.png"
    roc_fig_path = figures_dir / f"roc_{exp_name}.png"

    pretty_name = prettify_experiment_name(exp_name)

    save_confusion_matrix_figure(
        cm_array,
        cm_fig_path,
        title=f"Confusion Matrix: {pretty_name}"
    )

    save_roc_curve_figure(
        y_true,
        y_prob,
        metrics["auc"],
        roc_fig_path,
        title=f"ROC Curve: {pretty_name}"
    )

    print("\n=== Test Results ===")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Sensitivity      : {metrics['sensitivity']:.4f}")
    print(f"Specificity      : {metrics['specificity']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1-score         : {metrics['f1']:.4f}")
    print(f"AUC              : {metrics['auc']:.4f}")
    print(f"PR-AUC           : {metrics['pr_auc']:.4f}")

    print("\n=== Confusion Matrix ===")
    print(np.array(metrics["confusion_matrix"]))
    print(
        f"TN: {metrics['tn']}, FP: {metrics['fp']}, "
        f"FN: {metrics['fn']}, TP: {metrics['tp']}"
    )

    print(f"\nSaved metrics     : {metrics_json_path}")
    print(f"Saved predictions : {predictions_path}")
    print(f"Saved CM figure   : {cm_fig_path}")
    print(f"Saved ROC figure  : {roc_fig_path}")


if __name__ == "__main__":
    main()