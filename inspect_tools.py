import pickle
from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/dataset_before_splitting.pkl")


print("=" * 70)
print("Loading dataset...")
print("=" * 70)

with open(DATASET_PATH, "rb") as f:
    df = pickle.load(f)

print()
print("Dataset shape:", df.shape)

print()
print("=" * 70)
print("TOOL DISTRIBUTION")
print("=" * 70)

print(df["tool"].value_counts(dropna=False))


print()
print("=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isna().sum())


print()
print("=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


print()
print("=" * 70)
print("REVIEW LENGTH")
print("=" * 70)

review_lengths = df["review"].fillna("").astype(str).str.len()

print("Minimum:", review_lengths.min())
print("Maximum:", review_lengths.max())
print("Average:", round(review_lengths.mean(), 2))


print()
print("=" * 70)
print("RATE DISTRIBUTION")
print("=" * 70)

print(df["rate"].value_counts(dropna=False).sort_index())


print()
print("=" * 70)
print("SAMPLE FOR EACH TOOL")
print("=" * 70)

for tool in df["tool"].dropna().unique():

    print()
    print("-" * 70)
    print("TOOL:", tool)
    print("-" * 70)

    sample = df[df["tool"] == tool].iloc[0]

    print("Index:", sample["Index"])
    print("Review:", sample["review"])
    print("Begin line:", sample["beginline"])
    print("End line:", sample["endline"])
    print("Rate:", sample["rate"])
    print("Prompt length:", sample["prompt_length"])


print()
print("=" * 70)
print("Inspection completed")
print("=" * 70)