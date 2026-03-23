from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Paths

input_path = Path("data/processed/ham10000_mel_nv_metadata.csv")
split_dir = Path("data/splits")
split_dir.mkdir(parents=True, exist_ok=True)


# 2. Load processed binary metadata

df = pd.read_csv(input_path)

print("=== Full binary metadata ===")
print(df.shape)
print(df["dx"].value_counts())


# 3. Create lesion-level table
# One lesion_id should belong to one diagnosis only

lesion_df = (
    df.groupby("lesion_id")
    .agg(
        dx=("dx", "first"),
        label=("label", "first"),
        num_images=("image_id", "count")
    )
    .reset_index()
)

print("\n=== Lesion-level table ===")
print(lesion_df.shape)
print(lesion_df["dx"].value_counts())

# Safety check
problem_lesions = df.groupby("lesion_id")["dx"].nunique()
problem_lesions = problem_lesions[problem_lesions > 1]

print("\n=== Lesions with more than one diagnosis ===")
print(len(problem_lesions))

if len(problem_lesions) > 0:
    print("WARNING: Some lesion_id values have multiple diagnoses!")
    print(problem_lesions.head())


# 4. Split lesion IDs
# Goal:
# train = 70%
# val   = 10%
# test  = 20%

# First: train (70%) vs temp (30%)
train_lesions, temp_lesions = train_test_split(
    lesion_df,
    test_size=0.30,
    random_state=42,
    stratify=lesion_df["label"]
)

# Second: split temp into val (10%) and test (20%)
# temp is 30% total, so val should be 1/3 of temp, test should be 2/3 of temp
val_lesions, test_lesions = train_test_split(
    temp_lesions,
    test_size=2/3,
    random_state=42,
    stratify=temp_lesions["label"]
)

print("\n=== Lesion split sizes ===")
print("Train lesions:", len(train_lesions))
print("Val lesions:", len(val_lesions))
print("Test lesions:", len(test_lesions))


# 5. Map lesion split back to image-level rows

train_ids = set(train_lesions["lesion_id"])
val_ids = set(val_lesions["lesion_id"])
test_ids = set(test_lesions["lesion_id"])

train_df = df[df["lesion_id"].isin(train_ids)].copy()
val_df = df[df["lesion_id"].isin(val_ids)].copy()
test_df = df[df["lesion_id"].isin(test_ids)].copy()


# 6. Sanity checks

overlap_train_val = train_ids.intersection(val_ids)
overlap_train_test = train_ids.intersection(test_ids)
overlap_val_test = val_ids.intersection(test_ids)

print("\n=== Overlap checks ===")
print("Train ∩ Val:", len(overlap_train_val))
print("Train ∩ Test:", len(overlap_train_test))
print("Val ∩ Test:", len(overlap_val_test))


# 7. Print class distribution

def show_stats(name, split_df):
    print(f"\n=== {name} IMAGE-LEVEL stats ===")
    print("Rows:", len(split_df))
    print(split_df["dx"].value_counts())
    print("Unique lesions:", split_df["lesion_id"].nunique())

show_stats("TRAIN", train_df)
show_stats("VAL", val_df)
show_stats("TEST", test_df)


# 8. Save split CSVs

train_path = split_dir / "train.csv"
val_path = split_dir / "val.csv"
test_path = split_dir / "test.csv"

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nSaved files:")
print(train_path)
print(val_path)
print(test_path)