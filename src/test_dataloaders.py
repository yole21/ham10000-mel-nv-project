from dataset import create_dataloaders


def main():
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=8,
        num_workers=0
    )

    print("=== Number of batches ===")
    print("Train:", len(train_loader))
    print("Val:", len(val_loader))
    print("Test:", len(test_loader))

    train_images, train_labels = next(iter(train_loader))

    print("\n=== Train batch ===")
    print("Images shape:", train_images.shape)
    print("Labels shape:", train_labels.shape)
    print("Labels:", train_labels)

    val_images, val_labels = next(iter(val_loader))

    print("\n=== Val batch ===")
    print("Images shape:", val_images.shape)
    print("Labels shape:", val_labels.shape)
    print("Labels:", val_labels)

    test_images, test_labels = next(iter(test_loader))

    print("\n=== Test batch ===")
    print("Images shape:", test_images.shape)
    print("Labels shape:", test_labels.shape)
    print("Labels:", test_labels)


if __name__ == "__main__":
    main()