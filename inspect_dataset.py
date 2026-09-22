import pickle
from pathlib import Path

dataset_path = Path("data/dataset_before_splitting.pkl")

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

with open(dataset_path, "rb") as f:
    data = pickle.load(f)

print()
print("Dataset type:")
print(type(data))

print()
print("Dataset information:")

if hasattr(data, "shape"):
    print("Shape:", data.shape)

if hasattr(data, "columns"):
    print("Columns:")
    print(list(data.columns))

if hasattr(data, "keys"):
    try:
        print("Keys:")
        print(list(data.keys()))
    except Exception:
        pass

try:
    print("Length:", len(data))
except Exception:
    pass

print()
print("=" * 60)
print("First item")
print("=" * 60)

try:
    first_item = data.iloc[0] if hasattr(data, "iloc") else data[0]
    print(first_item)
except Exception as e:
    print("Could not display first item:", e)

print()
print("=" * 60)
print("Inspection completed")
print("=" * 60)