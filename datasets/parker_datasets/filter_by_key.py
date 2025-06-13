import argparse
import pandas as pd
import numpy as np


def filter_by_key(
    gold_standard_path,
    corrupted_dataset_path,
    primary_key,
    n=1000,
    out_gold="cleaned_gold_first1000.csv",
    out_corrupt="cleaned_corrupted_first1000.csv",
):
    # Read datasets
    corrupt_df = pd.read_csv(corrupted_dataset_path)
    gold_df = pd.read_csv(gold_standard_path)

    # Remove quotes from primary_key column if present
    corrupt_df[primary_key] = corrupt_df[primary_key].astype(str).str.replace('"', "")
    gold_df[primary_key] = gold_df[primary_key].astype(str).str.replace('"', "")

    # Find intersection of keys
    common_keys = set(corrupt_df[primary_key]).intersection(set(gold_df[primary_key]))

    # Filter both DataFrames to only those keys
    cleaned_corrupt = corrupt_df[corrupt_df[primary_key].isin(common_keys)]
    cleaned_gold = gold_df[gold_df[primary_key].isin(common_keys)]

    # Take first N unique keys from cleaned_corrupt
    cleaned_corrupt_firstn_keys = list(cleaned_corrupt[primary_key].unique()[:n])
    cleaned_corrupt_firstn = (
        cleaned_corrupt[cleaned_corrupt[primary_key].isin(cleaned_corrupt_firstn_keys)]
        .sort_values(by=primary_key)
        .reset_index(drop=True)
    )
    # Reorder columns to match gold standard
    cleaned_corrupt_firstn = cleaned_corrupt_firstn[gold_df.columns]

    # Reorder columns to ensure primary key is first, then the rest in original order (excluding primary key)
    def reorder_columns(df, primary_key):
        cols = [primary_key] + [col for col in df.columns if col != primary_key]
        return df[cols]

    cleaned_corrupt_firstn = reorder_columns(cleaned_corrupt_firstn, primary_key)
    cleaned_corrupt_firstn.to_csv(out_corrupt, index=False)

    cleaned_gold_firstn = (
        cleaned_gold[cleaned_gold[primary_key].isin(cleaned_corrupt_firstn_keys)]
        .sort_values(by=primary_key)
        .reset_index(drop=True)
    )
    cleaned_gold_firstn = cleaned_gold_firstn[gold_df.columns]
    cleaned_gold_firstn = reorder_columns(cleaned_gold_firstn, primary_key)
    cleaned_gold_firstn.to_csv(out_gold, index=False)

    # Ensure both DataFrames have the same primary key values in the same order
    try:
        assert list(cleaned_gold_firstn[primary_key].values) == list(
            cleaned_corrupt_firstn[primary_key].values
        ), "Primary key values do not match or are not in the same order"
        print("[CHECK PASSED] Primary key values match and are in the same order.")
    except AssertionError as e:
        print(f"[CHECK FAILED] {e}")
        raise

    # Additional check: ensure the number of rows is the same
    try:
        assert len(cleaned_gold_firstn) == len(
            cleaned_corrupt_firstn
        ), "Row count does not match between gold and corrupted datasets!"
        print("[CHECK PASSED] Row count matches between gold and corrupted datasets.")
    except AssertionError as e:
        print(f"[CHECK FAILED] {e}")
        raise

    # Print statistics about the corruption and differences
    print("\n--- Dataset Statistics ---")
    print(f"Original corrupted dataset rows: {len(corrupt_df)}")
    print(f"Original gold standard rows: {len(gold_df)}")
    print(f"Number of common keys: {len(common_keys)}")
    print(f"Rows in cleaned_corrupt: {len(cleaned_corrupt)}")
    print(f"Rows in cleaned_gold: {len(cleaned_gold)}")

    # Keys only in corrupted dataset
    corrupt_only_keys = set(corrupt_df[primary_key]) - set(gold_df[primary_key])
    print(f"Keys only in corrupted dataset: {len(corrupt_only_keys)}")

    # Keys only in gold standard
    gold_only_keys = set(gold_df[primary_key]) - set(corrupt_df[primary_key])
    print(f"Keys only in gold standard: {len(gold_only_keys)}")

    num_diffs = (
        (cleaned_corrupt_firstn.values != cleaned_gold_firstn.values).sum().sum()
    )
    print(f"Number of differences in first {n} rows: {num_diffs}")

    # Calculate and print percentage of values that are not the same
    total_values = cleaned_corrupt_firstn.shape[0] * cleaned_corrupt_firstn.shape[1]
    percent_diff = (num_diffs / total_values) * 100 if total_values else 0
    print(f"Percentage of values not the same in first {n} rows: {percent_diff:.2f}%\n")

    unique_primary_keys = cleaned_corrupt_firstn[primary_key].unique()
    print(f"Unique primary keys in cleaned_corrupt_firstn: {len(unique_primary_keys)}")

    print(f"Saved {len(cleaned_corrupt_firstn)} rows to {out_corrupt}")
    print(f"Saved {len(cleaned_gold_firstn)} rows to {out_gold}")
