from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score

from dataset import create_dataloaders
from model import create_resnet18_model


# ============================================================
# train_experiment.py
# Main experiment training script for HAM10000 MEL vs NV
#
# Experimental design:
#   1) Freeze + CE
#   2) Freeze + Weighted CE
#   3) Fine-tune + CE
#   4) Fine-tune + Weighted CE
#
# This version is matched to your current project structure:
#   - dataset.py: create_dataloaders(...)
#   - model.py  : create_resnet18_model(...)
#
# Current dataset loader returns:
#   (image, label)
# So this script focuses on robust training + validation logging.
# ============================================================


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Arguments
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train main experiment model for HAM10000 MEL vs NV")

    parser.add_argument("--mode", type=str, choices=["freeze", "finetune"], required=True,
                        help="Transfer learning mode")
    parser.add_argument("--loss_type", type=str, choices=["ce", "weighted"], required=True,
                        help="Loss function type")

    parser.add_argument("--train_csv", type=str, default="data/splits/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/splits/val.csv")
    parser.add_argument("--test_csv", type=str, default="data/splits/test.csv")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--save_metric", type=str, choices=["val_loss", "val_auc"], default="val_loss")
    parser.add_argument("--early_stopping_patience", type=int, default=5)

    return parser.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def get_experiment_name(mode: str, loss_type: str, seed: int) -> str:
    return f"{mode}_{loss_type}_seed{seed}"



def make_output_dirs(base_output_dir: str):
    base = Path(base_output_dir)
    paths = {
        "base": base,
        "models": base / "models",
        "metrics": base / "metrics",
        "logs": base / "logs",
        "configs": base / "configs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths



def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# -----------------------------
# Class weights for Weighted CE
# -----------------------------
def compute_class_weights_from_train_csv(train_csv: str, device: torch.device) -> torch.Tensor:
    import pandas as pd

    df = pd.read_csv(train_csv)
    if "label" not in df.columns:
        raise ValueError(f"'label' column not found in {train_csv}")

    labels = df["label"].astype(int).to_numpy()
    num_neg = int((labels == 0).sum())
    num_pos = int((labels == 1).sum())

    if num_neg == 0 or num_pos == 0:
        raise ValueError("Both classes must exist in the training set for weighted CE.")

    total = num_neg + num_pos
    weight_neg = total / (2.0 * num_neg)
    weight_pos = total / (2.0 * num_pos)

    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32, device=device)
    return class_weights



def build_criterion(loss_type: str, train_csv: str, device: torch.device):
    if loss_type == "ce":
        return nn.CrossEntropyLoss()

    if loss_type == "weighted":
        class_weights = compute_class_weights_from_train_csv(train_csv, device)
        print(f"Using weighted CE with class weights: {class_weights.detach().cpu().tolist()}")
        return nn.CrossEntropyLoss(weight=class_weights)

    raise ValueError("loss_type must be either 'ce' or 'weighted'")


# -----------------------------
# Metrics
# -----------------------------
def collect_probs_and_labels(outputs, labels):
    probs = torch.softmax(outputs, dim=1)[:, 1]
    return probs.detach().cpu().numpy(), labels.detach().cpu().numpy()



def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return metrics


# -----------------------------
# Train / validation loops
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs, true_labels = collect_probs_and_labels(outputs, labels)
        all_probs.extend(probs.tolist())
        all_labels.extend(true_labels.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_probs, threshold=0.5)
    return epoch_loss, metrics



def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs, true_labels = collect_probs_and_labels(outputs, labels)
            all_probs.extend(probs.tolist())
            all_labels.extend(true_labels.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_classification_metrics(all_labels, all_probs, threshold=0.5)
    return epoch_loss, metrics


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    freeze_backbone = True if args.mode == "freeze" else False
    exp_name = get_experiment_name(args.mode, args.loss_type, args.seed)
    output_dirs = make_output_dirs(args.output_dir)

    config = {
        "experiment_name": exp_name,
        "mode": args.mode,
        "loss_type": args.loss_type,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "save_metric": args.save_metric,
        "early_stopping_patience": args.early_stopping_patience,
        "model_name": "resnet18",
        "freeze_backbone": freeze_backbone,
    }
    save_json(config, output_dirs["configs"] / f"{exp_name}.json")

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    else:
        print("GPU not available, using CPU.")

    print("Experiment:", exp_name)

    # 2. Data
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    # 3. Model
    model = create_resnet18_model(num_classes=2, freeze_backbone=freeze_backbone)
    model = model.to(device)

    # 4. Loss and optimizer
    criterion = build_criterion(args.loss_type, args.train_csv, device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    # 5. Training loop
    best_score = float("inf") if args.save_metric == "val_loss" else -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history = []

    save_path = output_dirs["models"] / f"{exp_name}.pth"
    metrics_path = output_dirs["metrics"] / f"{exp_name}.json"
    history_path = output_dirs["logs"] / f"{exp_name}_history.json"

    train_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_metrics = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        current_score = val_loss if args.save_metric == "val_loss" else val_metrics["auc"]
        is_better = current_score < best_score if args.save_metric == "val_loss" else current_score > best_score

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - epoch_start, 2),
        })

        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Sens: {train_metrics['sensitivity']:.4f} | "
            f"Spec: {train_metrics['specificity']:.4f} | "
            f"F1: {train_metrics['f1']:.4f} | "
            f"AUC: {train_metrics['auc']:.4f}"
        )
        print(
            f"Val   Loss: {val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"Sens: {val_metrics['sensitivity']:.4f} | "
            f"Spec: {val_metrics['specificity']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        if is_better:
            best_score = current_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "best_score": best_score,
                "save_metric": args.save_metric,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            torch.save(checkpoint, save_path)
            print(f"Saved best model to: {save_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping triggered after {args.early_stopping_patience} epochs without improvement."
            )
            break

    total_train_time = round(time.time() - train_start_time, 2)

    save_json({
        "experiment_name": exp_name,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "save_metric": args.save_metric,
        "total_train_seconds": total_train_time,
        "history": history,
    }, history_path)

    save_json({
        "experiment_name": exp_name,
        "best_model_path": str(save_path),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "save_metric": args.save_metric,
        "final_val_metrics": history[best_epoch - 1]["val_metrics"] if best_epoch > 0 else None,
        "config": config,
    }, metrics_path)

    print("\nTraining finished.")
    print(f"Best epoch   : {best_epoch}")
    print(f"Saved model  : {save_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
