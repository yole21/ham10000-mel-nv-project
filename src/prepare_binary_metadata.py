from pathlib import Path
import pandas as pd

# 1. Paths

metadata_path = Path("data/raw/HAM10000_metadata.csv")
img_dir_1 = Path("data/raw/HAM10000_images_part_1")
img_dir_2 = Path("data/raw/HAM10000_images_part_2")

# Output path

output_path = Path("data/processed/ham10000_mel_nv_metadata.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)


# 2. Load metadata

df = pd.read_csv(metadata_path)

print("=== Original metadata shape ===")
print(df.shape)


# 3. Keep only mel and nv

df_binary = df[df["dx"].isin(["mel", "nv"])].copy()

print("\n=== Binary metadata shape (mel + nv only) ===")
print(df_binary.shape)

print("\n=== Binary class counts ===")
print(df_binary["dx"].value_counts())

print("\n=== Unique lesion_id count ===")
print(df_binary["lesion_id"].nunique())

print("\n=== Unique image_id count ===")
print(df_binary["image_id"].nunique())


# 4. Create label column
# mel = 1, nv = 0

df_binary["label"] = df_binary["dx"].map({"nv": 0, "mel": 1})

print("\n=== Label counts ===")
print(df_binary["label"].value_counts())


# 5. Build image path

def find_image_path(image_id: str) -> str:
    filename = f"{image_id}.jpg"

    path1 = img_dir_1 / filename
    if path1.exists():
        return str(path1)

    path2 = img_dir_2 / filename
    if path2.exists():
        return str(path2)

    return ""

df_binary["image_path"] = df_binary["image_id"].apply(find_image_path)


# 6. Check missing image paths

missing_count = (df_binary["image_path"] == "").sum()

print("\n=== Missing image path count ===")
print(missing_count)

if missing_count > 0:
    print("\nWARNING: Some images were not found.")
    print(df_binary[df_binary["image_path"] == ""][["image_id", "dx"]].head())
else:
    print("\nAll MEL/NV image paths were found successfully.")


# 7. Save processed binary metadata

df_binary.to_csv(output_path, index=False)

print(f"\nSaved processed file to: {output_path}")
print("\n=== First 5 rows of processed binary metadata ===")
print(df_binary.head())