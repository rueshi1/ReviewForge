import pickle
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/dataset_before_splitting.pkl")
OUTPUT_FILE = Path("filtered_data_token_length.csv")


print("=" * 70)
print("CONVERTING DATASET")
print("=" * 70)

print()
print("Input :", INPUT_FILE)
print("Output:", OUTPUT_FILE)

print()
print("Loading dataset...")

with open(INPUT_FILE, "rb") as f:
    df = pickle.load(f)

print("Loaded successfully.")
print("Shape:", df.shape)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Saving CSV...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("CSV created successfully.")

print()
print("File size:")

size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)

print(f"{size_mb:.2f} MB")

print()
print("=" * 70)
print("CONVERSION COMPLETED")
print("=" * 70)