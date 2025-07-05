import numpy as np
import pandas as pd


def precision(true_positive: int, false_positive: int) -> float:
    """
    Calculate precision.
    """
    if true_positive + false_positive == 0:
        return 0.0
    return true_positive / (true_positive + false_positive)


def recall(true_positive: int, false_negative: int) -> float:
    """
    Calculate recall.
    """
    if true_positive + false_negative == 0:
        return 0.0
    return true_positive / (true_positive + false_negative)


def f1_score(true_positive: int, false_positive: int, false_negative: int) -> float:
    """
    Calculate F1 score from true positives, false positives, and false negatives.
    """
    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def accuracy(
    true_positive: int, true_negative: int, false_positive: int, false_negative: int
) -> float:
    """
    Calculate accuracy.
    """
    total = true_positive + true_negative + false_positive + false_negative
    if total == 0:
        return 0.0
    return (true_positive + true_negative) / total


def false_positive_rate(false_positive: int, true_negative: int) -> float:
    """
    Calculate the false positive rate (FPR).
    """
    if false_positive + true_negative == 0:
        return 0.0
    return false_positive / (false_positive + true_negative)


def false_negative_rate(false_negative: int, true_positive: int) -> float:
    """
    Calculate the false negative rate (FNR).
    """
    if false_negative + true_positive == 0:
        return 0.0
    return false_negative / (false_negative + true_positive)


def calculate_stats(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict:
    prec = precision(true_positive=true_positive, false_positive=false_positive)
    rec = recall(true_positive=true_positive, false_negative=false_negative)
    f1 = f1_score(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    acc = accuracy(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    fpr = false_positive_rate(
        false_positive=false_positive, true_negative=true_negative
    )
    fnr = false_negative_rate(
        false_negative=false_negative, true_positive=true_positive
    )
    stats = {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "accuracy": acc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }

    return stats


def calculate_macro_stats(macro_stats: dict) -> dict:
    """
    Calculate macro statistics from a list of column-wise statistics.
    """
    stats = {
        "num_rows": macro_stats["num_rows"],
        "num_columns": macro_stats["num_columns"],
        "column_names": macro_stats["column_names"],
    }
    prec = np.mean([col["precision"] for col in macro_stats["stats"]])
    rec = np.mean([col["recall"] for col in macro_stats["stats"]])
    f1 = np.mean([col["f1_score"] for col in macro_stats["stats"]])
    acc = np.mean([col["accuracy"] for col in macro_stats["stats"]])
    fpr = np.mean([col["false_positive_rate"] for col in macro_stats["stats"]])
    fnr = np.mean([col["false_negative_rate"] for col in macro_stats["stats"]])

    stats.update(
        {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "accuracy": acc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        }
    )

    return stats


def preprocess(
    gold_standard: pd.DataFrame,
    cleaned_dataset: pd.DataFrame,
    original_dataset: pd.DataFrame,
    primary_key: str,
):
    # Check columns of all
    columns_list = [
        set(df.columns) for df in [gold_standard, cleaned_dataset, original_dataset]
    ]
    if not all(cols == columns_list[0] for cols in columns_list):
        raise ValueError(
            f"All input DataFrames must have the same columns.\n"
            f"gold_standard columns: {list(gold_standard.columns)}\n"
            f"cleaned_dataset columns: {list(cleaned_dataset.columns)}\n"
            f"original_dataset columns: {list(original_dataset.columns)}"
        )

    common_keys = list(
        set(gold_standard[primary_key])
        & set(original_dataset[primary_key])
        & set(cleaned_dataset[primary_key])
    )

    rows_from_cleand_not_in_common = cleaned_dataset[
        ~cleaned_dataset[primary_key].isin(common_keys)
    ]

    gold_standard = gold_standard[
        gold_standard[primary_key].isin(common_keys)
    ].reset_index(drop=True)
    cleaned_dataset = cleaned_dataset[
        cleaned_dataset[primary_key].isin(common_keys)
    ].reset_index(drop=True)
    original_dataset = original_dataset[
        original_dataset[primary_key].isin(common_keys)
    ].reset_index(drop=True)

    # Sort all datasets by primary key and reset index to ensure same row order
    gold_standard = gold_standard.sort_values(by=primary_key).reset_index(drop=True)
    cleaned_dataset = cleaned_dataset.sort_values(by=primary_key).reset_index(drop=True)
    original_dataset = original_dataset.sort_values(by=primary_key).reset_index(
        drop=True
    )

    # Ensure same column order (already done above, but repeat for safety)
    columns_order = gold_standard.columns
    gold_standard = gold_standard[columns_order]
    cleaned_dataset = cleaned_dataset[columns_order]
    original_dataset = original_dataset[columns_order]

    # Assert that all datasets have the same column order
    assert (
        list(gold_standard.columns)
        == list(cleaned_dataset.columns)
        == list(original_dataset.columns)
    ), "All datasets must have the same column order."

    # Assert that all datasets have the same row order for the primary key
    if not (
        list(gold_standard[primary_key])
        == list(cleaned_dataset[primary_key])
        == list(original_dataset[primary_key])
    ):
        raise ValueError(
            "All datasets must have the same row order for the primary key.\n"
            f"gold_standard[{primary_key}]: {list(gold_standard[primary_key])}\n"
            f"cleaned_dataset[{primary_key}]: {list(cleaned_dataset[primary_key])}\n"
            f"original_dataset[{primary_key}]: {list(original_dataset[primary_key])}"
        )

    return (
        gold_standard,
        cleaned_dataset,
        original_dataset,
        rows_from_cleand_not_in_common,
    )


def evaluate_dataset_micro(
    gold_standard: pd.DataFrame,
    cleaned_dataset: pd.DataFrame,
    original_dataset: pd.DataFrame,
    primary_key: str,
) -> dict:
    """
    Evaluate the cleaned dataset against a gold standard dataset using micro-averaged metrics.

    Args:
        gold_standard (pd.DataFrame): The ground truth dataset.
        cleaned_dataset (pd.DataFrame): The cleaned dataset to evaluate.
        corrupted_dataset (pd.DataFrame): The corrupted dataset (before cleaning).
        primary_keys (list[str]): List of column names that are primary keys and should be ignored in evaluation.

    Returns:
        dict: A dictionary containing overall evaluation metrics.
    """

    gold_standard, cleaned_dataset, original_dataset, rows_from_cleaned_not_in_gold = (
        preprocess(gold_standard, cleaned_dataset, original_dataset, primary_key)
    )

    if not (gold_standard.shape == cleaned_dataset.shape == original_dataset.shape):
        raise ValueError("Datasets must have the same shape for evaluation.")

    gold_standard = gold_standard.astype(str)
    cleaned_dataset = cleaned_dataset.astype(str)
    original_dataset = original_dataset.astype(str)

    n_rows, n_cols = gold_standard.shape
    stats = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "column_names": list(gold_standard.columns),
    }

    true_positive = 0
    false_positive = (
        rows_from_cleaned_not_in_gold.shape[0] * rows_from_cleaned_not_in_gold.shape[1]
        if rows_from_cleaned_not_in_gold is not None
        else 0
    )
    false_negative = 0
    true_negative = 0

    for row in range(gold_standard.shape[0]):
        for col in cleaned_dataset.columns:
            if col in primary_key:
                continue  # Skip primary key columns

            is_corrupted = gold_standard.at[row, col] != original_dataset.at[row, col]
            gold_val = gold_standard.at[row, col]
            clean_val = cleaned_dataset.at[row, col]
            # True positive: cell was corrupted and correctly fixed
            if is_corrupted and gold_val == clean_val:
                true_positive += 1
            # False positive: cell was not corrupted but was changed incorrectly
            elif not is_corrupted and gold_val != clean_val:
                false_positive += 1
            # False negative: cell was corrupted but not fixed (FP bei Parker-repoducebility)
            elif is_corrupted and gold_val != clean_val:
                false_negative += 1
            # True negative: cell was not corrupted and not changed
            elif not is_corrupted and gold_val == clean_val:
                true_negative += 1

    # Calculate metrics
    stats.update(
        calculate_stats(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
        )
    )

    return stats


def evaluate_dataset_macro(
    gold_standard: pd.DataFrame,
    cleaned_dataset: pd.DataFrame,
    original_dataset: pd.DataFrame,
    primary_key: str,
) -> dict:
    """
    Evaluate the cleaned dataset against a gold standard dataset using macro-averaged metrics (per column).

    Args:
        gold_standard (pd.DataFrame): The ground truth dataset.
        cleaned_dataset (pd.DataFrame): The cleaned dataset to evaluate.
        corrupted_dataset (pd.DataFrame): The corrupted dataset (before cleaning).
        primary_keys (list[str]): List of column names that are primary keys and should be ignored in evaluation.

    Returns:
        dict: A dictionary containing evaluation metrics per column.
    """

    gold_standard, cleaned_dataset, original_dataset, rows_from_cleaned_not_in_gold = (
        preprocess(gold_standard, cleaned_dataset, original_dataset, primary_key)
    )

    if not (gold_standard.shape == cleaned_dataset.shape == original_dataset.shape):
        raise ValueError("Datasets must have the same shape for evaluation.")

    gold_standard = gold_standard.astype(str)
    cleaned_dataset = cleaned_dataset.astype(str)
    original_dataset = original_dataset.astype(str)

    n_rows, n_cols = gold_standard.shape
    stats_per_column = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "column_names": list(gold_standard.columns),
        "stats": [],
    }

    # Evaluate each column separately, skipping primary key columns
    for col in cleaned_dataset.columns:
        if col in primary_key:
            continue  # Skip primary key columns

        true_positive = 0
        false_positive = (
            rows_from_cleaned_not_in_gold.shape[0]
            if rows_from_cleaned_not_in_gold is not None
            else 0
        )
        false_negative = 0
        true_negative = 0
        for row in range(n_rows):
            is_corrupted = gold_standard.at[row, col] != original_dataset.at[row, col]
            gold_val = gold_standard.at[row, col]
            clean_val = cleaned_dataset.at[row, col]
            # True positive: cell was corrupted and correctly fixed
            if is_corrupted and gold_val == clean_val:
                true_positive += 1
            # False positive: cell was not corrupted but was changed incorrectly
            elif not is_corrupted and gold_val != clean_val:
                false_positive += 1
            # False negative: cell was corrupted but not fixed
            elif is_corrupted and gold_val != clean_val:
                false_negative += 1
            # True negative: cell was not corrupted and not changed
            elif not is_corrupted and gold_val == clean_val:
                true_negative += 1
        col_stats = {
            "num_entries": n_rows,
            "column_name": col,
        }
        col_stats.update(
            calculate_stats(
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                true_negative=true_negative,
            )
        )
        stats_per_column["stats"].append(col_stats)
    return stats_per_column
