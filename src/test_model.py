import torch
from model import create_resnet18_model, count_total_parameters, count_trainable_parameters


def main():
    freeze_model = create_resnet18_model(num_classes=2, freeze_backbone=True)
    finetune_model = create_resnet18_model(num_classes=2, freeze_backbone=False)

    print("=== Freeze model ===")
    print("Total parameters:", count_total_parameters(freeze_model))
    print("Trainable parameters:", count_trainable_parameters(freeze_model))

    print("\n=== Fine-tune model ===")
    print("Total parameters:", count_total_parameters(finetune_model))
    print("Trainable parameters:", count_trainable_parameters(finetune_model))

    x = torch.randn(8, 3, 224, 224)

    freeze_out = freeze_model(x)
    finetune_out = finetune_model(x)

    print("\n=== Output shape check ===")
    print("Freeze output shape:", freeze_out.shape)
    print("Fine-tune output shape:", finetune_out.shape)


if __name__ == "__main__":
    main()