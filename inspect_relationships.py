import pickle
from pathlib import Path

DATASET_PATH = Path("data/dataset_before_splitting.pkl")

print("Loading dataset...")

with open(DATASET_PATH, "rb") as f:
    df = pickle.load(f)

print("Dataset shape:", df.shape)

print()
print("=" * 60)
print("TOOL DISTRIBUTION")
print("=" * 60)
print(df["tool"].value_counts())

print()
print("=" * 60)
print("UNIQUE INDEX COUNTS")
print("=" * 60)

for tool in ["PMD", "checkstyle", "CodeLlama"]:
    subset = df[df["tool"] == tool]

    print(
        f"{tool:12} "
        f"records = {len(subset):6} "
        f"unique Index = {subset['Index'].nunique():6}"
    )

print()
print("=" * 60)
print("INDEX OVERLAP")
print("=" * 60)

pmd = set(df.loc[df["tool"] == "PMD", "Index"])
checkstyle = set(df.loc[df["tool"] == "checkstyle", "Index"])
codellama = set(df.loc[df["tool"] == "CodeLlama", "Index"])

print("PMD ∩ Checkstyle :", len(pmd & checkstyle))
print("PMD ∩ CodeLlama  :", len(pmd & codellama))
print("Checkstyle ∩ CodeLlama:", len(checkstyle & codellama))
print("All three tools  :", len(pmd & checkstyle & codellama))

print()
print("=" * 60)
print("REVIEWS PER INDEX")
print("=" * 60)

reviews_per_index = df.groupby("Index").size()

print("Unique Index values :", len(reviews_per_index))
print("Minimum reviews     :", reviews_per_index.min())
print("Maximum reviews     :", reviews_per_index.max())
print("Average reviews     :", round(reviews_per_index.mean(), 2))

print()
print("=" * 60)
print("TOOL COMBINATIONS")
print("=" * 60)

combinations = (
    df.groupby("Index")["tool"]
      .apply(lambda x: "+".join(sorted(x.unique())))
      .value_counts()
)

print(combinations)

print()
print("=" * 60)
print("SAMPLE INDEXES")
print("=" * 60)

all_three = pmd & checkstyle & codellama

for index in sorted(all_three)[:10]:

    rows = df[df["Index"] == index]

    print()
    print(
        f"Index {index} | "
        f"tools = {', '.join(rows['tool'].unique())} | "
        f"records = {len(rows)}"
    )

    for _, row in rows.iterrows():

        review = str(row["review"])

        print(
            f"  {row['tool']:12} "
            f"begin={row['beginline']} "
            f"end={row['endline']} "
            f"rate={row['rate']} "
            f"review_length={len(review)}"
        )

print()
print("=" * 60)
print("Inspection completed")
print("=" * 60)