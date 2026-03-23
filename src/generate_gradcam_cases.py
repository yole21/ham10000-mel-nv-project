import os
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models


# =========================
# 1. Build model
# =========================
def build_model(arch="resnet50", num_classes=2):
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "resnet34":
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif arch == "resnet101":
        model = models.resnet101(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported arch: {arch}")

    return model


# =========================
# 2. Load checkpoint
# =========================
def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Common cases:
    # 1) pure state_dict
    # 2) dict with "model_state_dict"
    # 3) dict with "state_dict"
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # maybe checkpoint itself is already a state dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove possible "module." prefix
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[len("module."):]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# =========================
# 3. Grad-CAM helper
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(self, input_tensor, target_class):
        output = self.model(input_tensor)
        self.model.zero_grad()

        score = output[:, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]          # [C, H, W]
        activations = self.activations[0]      # [C, H, W]

        weights = gradients.mean(dim=(1, 2))   # [C]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, output.detach()


# =========================
# 4. Utilities
# =========================
def get_target_layer(model, arch):
    # Good default target layers for torchvision ResNet
    if arch in ["resnet18", "resnet34"]:
        return model.layer4[-1].conv2
    elif arch in ["resnet50", "resnet101"]:
        return model.layer4[-1].conv3
    else:
        raise ValueError(f"Unsupported arch for target layer: {arch}")


def preprocess_image(image_path, image_size=224):
    pil_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(pil_img)

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = transform(pil_img).unsqueeze(0)
    return pil_img, orig_np, input_tensor


def make_heatmap(cam, width, height):
    cam_resized = cv2.resize(cam, (width, height))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap_on_image(image_rgb, heatmap_rgb, alpha=0.4):
    overlay = np.clip((1 - alpha) * image_rgb + alpha * heatmap_rgb, 0, 255).astype(np.uint8)
    return overlay


def save_rgb(path, array_rgb):
    Image.fromarray(array_rgb).save(path)


# =========================
# 5. Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected_csv", type=str, required=True, help="CSV from select_error_cases.py")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--arch", type=str, default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--target_class", type=int, default=1, help="Melanoma class index (usually 1)")
    parser.add_argument("--out_dir", type=str, default="outputs/gradcam_cases")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(arch=args.arch, num_classes=2)
    model = load_checkpoint(model, args.model_path, device)

    target_layer = get_target_layer(model, args.arch)
    gradcam = GradCAM(model, target_layer)

    selected_df = pd.read_csv(args.selected_csv)
    summary_rows = []

    for idx, row in selected_df.iterrows():
        case_type = row["case_type"]
        image_id = row["image_id"]
        image_path = row["image_path"]
        true_label = int(row["true_label"])
        pred_label = int(row["pred_label"])
        saved_prob_mel = float(row["prob_mel"])

        if not os.path.exists(image_path):
            print(f"[WARNING] File not found: {image_path}")
            continue

        pil_img, orig_np, input_tensor = preprocess_image(image_path, image_size=args.image_size)
        input_tensor = input_tensor.to(device)

        cam, logits = gradcam.generate(input_tensor, target_class=args.target_class)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prob_mel = float(probs[args.target_class])

        h, w = orig_np.shape[:2]
        heatmap_rgb = make_heatmap(cam, w, h)
        overlay_rgb = overlay_heatmap_on_image(orig_np, heatmap_rgb, alpha=0.4)

        stem = f"{idx+1:02d}_{case_type}_{image_id}"

        orig_path = os.path.join(args.out_dir, f"{stem}_orig.png")
        heatmap_path = os.path.join(args.out_dir, f"{stem}_heatmap.png")
        overlay_path = os.path.join(args.out_dir, f"{stem}_overlay.png")

        save_rgb(orig_path, orig_np)
        save_rgb(heatmap_path, heatmap_rgb)
        save_rgb(overlay_path, overlay_rgb)

        summary_rows.append({
            "case_type": case_type,
            "image_id": image_id,
            "image_path": image_path,
            "true_label": true_label,
            "pred_label": pred_label,
            "saved_prob_mel": saved_prob_mel,
            "recomputed_prob_mel": prob_mel,
            "orig_path": orig_path,
            "heatmap_path": heatmap_path,
            "overlay_path": overlay_path,
        })

        print(f"Saved Grad-CAM for {case_type} | {image_id} | prob_mel={prob_mel:.4f}")

    gradcam.remove_hooks()

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.out_dir, "gradcam_case_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\nDone.")
    print(f"Saved outputs to: {args.out_dir}")
    print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()