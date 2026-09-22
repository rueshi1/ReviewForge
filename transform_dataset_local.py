import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


INPUT_FILE = "filtered_data_token_length.csv"
OUTPUT_DIR = "processed_dataset_new_prompt"


def transform_dataset():

    print("=" * 70)
    print("TRANSFORMING DATASET")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print(f"Total examples: {len(df)}")

    # ---------------------------------------------------------
    # 2. Remove NaN reviews
    # ---------------------------------------------------------

    nan_count = df["review"].isna().sum()

    if nan_count > 0:
        print(f"Found {nan_count} rows with NaN review.")
        df = df.dropna(subset=["review"])

    print(f"Rows after NaN removal: {len(df)}")

    # ---------------------------------------------------------
    # 3. Remove exact duplicate examples
    #
    # Same code + patch + review + tool
    # ---------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["code", "patch", "review", "tool"]
    ).reset_index(drop=True)

    duplicates_removed = before_duplicates - len(df)

    print(f"Exact duplicate rows removed: {duplicates_removed}")
    print(f"Rows after duplicate removal: {len(df)}")

    # ---------------------------------------------------------
    # 4. Create GROUP KEY
    #
    # IMPORTANT:
    # One code + patch = one group.
    #
    # All reviews/tools belonging to the same code change
    # MUST stay in the same split.
    # ---------------------------------------------------------

    df["group_key"] = (
        df["code"].fillna("").astype(str)
        + "|||"
        + df["patch"].fillna("").astype(str)
    )

    unique_groups = df["group_key"].nunique()

    print(f"Unique code+patch groups: {unique_groups}")

    group_sizes = df.groupby("group_key").size()

    print(f"Groups with multiple rows: {(group_sizes > 1).sum()}")
    print(f"Maximum rows in one group: {group_sizes.max()}")

    # ---------------------------------------------------------
    # 5. Get unique groups
    # ---------------------------------------------------------

    groups = pd.DataFrame({
        "group_key": df["group_key"].unique()
    })

    # ---------------------------------------------------------
    # 6. Split GROUPS
    #
    # 80% train
    # 10% validation
    # 10% test
    # ---------------------------------------------------------

    train_groups, temp_groups = train_test_split(
        groups,
        test_size=0.20,
        random_state=42
    )

    valid_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        random_state=42
    )

    train_group_set = set(train_groups["group_key"])
    valid_group_set = set(valid_groups["group_key"])
    test_group_set = set(test_groups["group_key"])

    print()
    print("=" * 70)
    print("GROUP SPLIT")
    print("=" * 70)

    print(f"Train groups:      {len(train_group_set)}")
    print(f"Validation groups: {len(valid_group_set)}")
    print(f"Test groups:       {len(test_group_set)}")

    # ---------------------------------------------------------
    # 7. Verify group split
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("GROUP LEAKAGE CHECK")
    print("=" * 70)

    print(
        "Train-Val overlap:",
        len(train_group_set & valid_group_set)
    )

    print(
        "Train-Test overlap:",
        len(train_group_set & test_group_set)
    )

    print(
        "Val-Test overlap:",
        len(valid_group_set & test_group_set)
    )

    assert len(train_group_set & valid_group_set) == 0
    assert len(train_group_set & test_group_set) == 0
    assert len(valid_group_set & test_group_set) == 0

    print("✓ No group leakage detected.")

    # ---------------------------------------------------------
    # 8. Create dataframe splits
    # ---------------------------------------------------------

    train = df[df["group_key"].isin(train_group_set)].copy()
    valid = df[df["group_key"].isin(valid_group_set)].copy()
    test = df[df["group_key"].isin(test_group_set)].copy()

    print()
    print("=" * 70)
    print("ROW COUNTS")
    print("=" * 70)

    print(f"Train rows:      {len(train)}")
    print(f"Validation rows: {len(valid)}")
    print(f"Test rows:       {len(test)}")
    print(f"Total rows:      {len(train) + len(valid) + len(test)}")

    # ---------------------------------------------------------
    # 9. Tool distribution
    # ---------------------------------------------------------

    print()
    print("Tool distribution:")

    for name, split in [
        ("Train", train),
        ("Validation", valid),
        ("Test", test)
    ]:
        print(f"\n{name}:")
        print(split["tool"].value_counts())

    # ---------------------------------------------------------
    # 10. Shuffle
    # ---------------------------------------------------------

    train = train.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    valid = valid.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    test = test.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # 11. Convert to HuggingFace datasets
    # ---------------------------------------------------------

    train_dataset = Dataset.from_pandas(
        train,
        preserve_index=False
    )

    valid_dataset = Dataset.from_pandas(
        valid,
        preserve_index=False
    )

    test_dataset = Dataset.from_pandas(
        test,
        preserve_index=False
    )

    dataset_splits = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset,
        "test": test_dataset
    })

    # ---------------------------------------------------------
    # 12. Create prompt/messages
    # ---------------------------------------------------------

    def process_example(example):

        text_user = (
            "Please analyze the following code change and provide a review, "
            "listing all potential issues, bugs, or areas for improvement. "
            "Be specific and comprehensive in your feedback.\n\n"
            "### Code Patch:\n"
            f"{example['patch']}\n\n"
            "### Review:"
        )

        text_assistant = example["review"]

        messages = [
            {
                "role": "user",
                "content": text_user
            },
            {
                "role": "assistant",
                "content": text_assistant
            }
        ]

        return {
            "messages": messages
        }

    print()
    print("=" * 70)
    print("CREATING MESSAGES")
    print("=" * 70)

    dataset_splits = dataset_splits.map(
        process_example,
        num_proc=1
    )

    # ---------------------------------------------------------
    # 13. FINAL LEAKAGE VERIFICATION
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL DATASET VERIFICATION")
    print("=" * 70)

    train_df = dataset_splits["train"].to_pandas()
    valid_df = dataset_splits["validation"].to_pandas()
    test_df = dataset_splits["test"].to_pandas()

    # Recreate group keys
    def get_keys(data):

        return set(
            zip(
                data["code"].fillna("").astype(str),
                data["patch"].fillna("").astype(str)
            )
        )

    train_keys = get_keys(train_df)
    valid_keys = get_keys(valid_df)
    test_keys = get_keys(test_df)

    train_valid_overlap = len(train_keys & valid_keys)
    train_test_overlap = len(train_keys & test_keys)
    valid_test_overlap = len(valid_keys & test_keys)

    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(valid_df)}")
    print(f"Test:       {len(test_df)}")

    print()
    print("Code+Patch overlap:")
    print(f"Train-Val:   {train_valid_overlap}")
    print(f"Train-Test:  {train_test_overlap}")
    print(f"Val-Test:    {valid_test_overlap}")

    assert train_valid_overlap == 0
    assert train_test_overlap == 0
    assert valid_test_overlap == 0

    print("✓ No code+patch leakage detected.")

    # ---------------------------------------------------------
    # 14. Check exact duplicates across splits
    # ---------------------------------------------------------

    all_df = pd.concat(
        [
            train_df,
            valid_df,
            test_df
        ],
        ignore_index=True
    )

    exact_duplicates = all_df.duplicated(
        subset=["code", "patch", "review", "tool"]
    ).sum()

    print()
    print(
        "Exact duplicate rows across final dataset:",
        exact_duplicates
    )

    # This should ideally be zero because we removed them before splitting.
    assert exact_duplicates == 0

    print("✓ No exact duplicate examples.")

    # ---------------------------------------------------------
    # 15. Save
    # ---------------------------------------------------------

    dataset_splits.save_to_disk(OUTPUT_DIR)

    print()
    print("=" * 70)
    print("DATASET SAVED")
    print("=" * 70)

    print(f"Location: {OUTPUT_DIR}")


if __name__ == "__main__":
    transform_dataset()