from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HAM10000Dataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    train_csv = Path("data/splits/train.csv")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = HAM10000Dataset(train_csv, transform=transform)

    print("=== Dataset basic info ===")
    print("Number of samples:", len(dataset))

    image, label = dataset[0]

    print("\n=== Single sample check ===")
    print("Image tensor shape:", image.shape)
    print("Image tensor dtype:", image.dtype)
    print("Label:", label)

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    batch_images, batch_labels = next(iter(loader))

    print("\n=== Batch check ===")
    print("Batch image shape:", batch_images.shape)
    print("Batch image dtype:", batch_images.dtype)
    print("Batch labels shape:", batch_labels.shape)
    print("Batch labels:", batch_labels)


if __name__ == "__main__":
    main()