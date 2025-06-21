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


def evaluate_dataset_micro(
    gold_standard: pd.DataFrame,
    cleaned_dataset: pd.DataFrame,
    corrupted_dataset: pd.DataFrame,
) -> dict:
    """
    Evaluate the generated dataset against a gold standard dataset.

    Args:
        gold_standard (pd.DataFrame): The ground truth dataset.
        generated_dataset (pd.DataFrame): The dataset to evaluate.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """

    if not (gold_standard.shape == cleaned_dataset.shape == corrupted_dataset.shape):
        raise ValueError("Datasets must have the same shape for evaluation.")

    gold_standard = gold_standard.astype(str)
    cleaned_dataset = cleaned_dataset.astype(str)
    corrupted_dataset = corrupted_dataset.astype(str)

    n_rows, n_cols = gold_standard.shape
    stats = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "column_names": list(gold_standard.columns),
    }

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for row in range(n_rows):
        for col in cleaned_dataset.columns:
            is_corrupted = gold_standard.at[row, col] != corrupted_dataset.at[row, col]
            gold_val = gold_standard.at[row, col]
            clean_val = cleaned_dataset.at[row, col]
            if is_corrupted and gold_val == clean_val:
                true_positive += 1
            elif not is_corrupted and gold_val != clean_val:
                false_positive += 1
            elif is_corrupted and gold_val != clean_val:
                false_negative += 1
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
    corrupted_dataset: pd.DataFrame,
) -> dict:
    """
    Evaluate the generated dataset against a gold standard dataset using macro metrics (per column).

    Args:
        gold_standard (pd.DataFrame): The ground truth dataset.
        cleaned_dataset (pd.DataFrame): The cleaned dataset to evaluate.
        corrupted_dataset (pd.DataFrame): The corrupted dataset (before cleaning).

    Returns:
        dict: A dictionary containing evaluation metrics per column.
    """
    if not (gold_standard.shape == cleaned_dataset.shape == corrupted_dataset.shape):
        raise ValueError("Datasets must have the same shape for evaluation.")

    gold_standard = gold_standard.astype(str)
    cleaned_dataset = cleaned_dataset.astype(str)
    corrupted_dataset = corrupted_dataset.astype(str)

    n_rows, n_cols = gold_standard.shape
    stats_per_column = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "column_names": list(gold_standard.columns),
        "stats": [],
    }

    for col in cleaned_dataset.columns:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        for row in range(n_rows):
            is_corrupted = gold_standard.at[row, col] != corrupted_dataset.at[row, col]
            gold_val = gold_standard.at[row, col]
            clean_val = cleaned_dataset.at[row, col]
            if is_corrupted and gold_val == clean_val:
                true_positive += 1
            elif not is_corrupted and gold_val != clean_val:
                false_positive += 1
            elif is_corrupted and gold_val != clean_val:
                false_negative += 1
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
