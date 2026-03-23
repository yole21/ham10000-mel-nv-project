from pathlib import Path
import pandas as pd

# 1. Set the metadata file path
metadata_path = Path("data/raw/HAM10000_metadata.csv")

# 2. Load the CSV file
df = pd.read_csv(metadata_path)

# 3. Print basic information
print("=== First 5 rows ===")
print(df.head())

print("\n=== Column names ===")
print(df.columns.tolist())

print("\n=== Diagnosis labels in dx ===")
print(sorted(df["dx"].unique().tolist()))

print("\n=== Class counts ===")
print(df["dx"].value_counts())

print("\n=== Number of rows ===")
print(len(df))

print("\n=== Number of unique lesion_id ===")
print(df["lesion_id"].nunique())

print("\n=== Number of unique image_id ===")
print(df["image_id"].nunique())