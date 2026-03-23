from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from dataset import create_dataloaders
from model import create_resnet18_model


def evaluate_model(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]   # probability of class 1
            preds = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    return all_labels, all_preds, all_probs


def main():
    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    # 2. Data
    batch_size = 32
    num_workers = 0

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 3. Model
    model = create_resnet18_model(num_classes=2, freeze_backbone=True)

    model_path = Path("outputs/models/finetune_weighted_ce.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    print(f"Loaded model from: {model_path}")

    # 4. Evaluation
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)

    # 5. Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)   # sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n=== Test Results ===")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Sensitivity : {recall:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1-score    : {f1:.4f}")
    print(f"AUC         : {auc:.4f}")

    print("\n=== Confusion Matrix ===")
    print(cm)
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")


if __name__ == "__main__":
    main()