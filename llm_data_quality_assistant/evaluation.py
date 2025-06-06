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
    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    f1 = f1_score(true_positive, false_positive, false_negative)
    acc = accuracy(true_positive, true_negative, false_positive, false_negative)
    fpr = false_positive_rate(false_positive, true_negative)
    fnr = false_negative_rate(false_negative, true_positive)
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
    generated_dataset: pd.DataFrame,
    corrupted_coords: list[np.ndarray],
) -> dict:
    """
    Evaluate the generated dataset against a gold standard dataset.

    Args:
        gold_standard (pd.DataFrame): The ground truth dataset.
        generated_dataset (pd.DataFrame): The dataset to evaluate.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    if gold_standard.shape != generated_dataset.shape:
        raise ValueError("Datasets must have the same shape for evaluation.")
    n_rows, n_cols = gold_standard.shape
    corrupted_coords_flattened = np.array(
        [coord for coords in corrupted_coords for coord in coords]
    )
    corrupted_coords_flattened = np.unique(corrupted_coords_flattened, axis=0)

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
        for col in range(n_cols):
            coord = np.array([row, col])
            is_corrupted = np.any(np.all(corrupted_coords_flattened == coord, axis=1))
            gold_val = gold_standard.iloc[row, col]
            gen_val = generated_dataset.iloc[row, col]
            if is_corrupted and gold_val == gen_val:
                true_positive += 1
            elif not is_corrupted and gold_val != gen_val:
                false_positive += 1
            elif is_corrupted and gold_val != gen_val:
                false_negative += 1
            elif not is_corrupted and gold_val == gen_val:
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
    generated_dataset: pd.DataFrame,
    corrupted_coords: list[np.ndarray],
) -> dict:
    """
    Evaluate the generated dataset against a gold standard dataset using macro metrics.

    Args:
        gold_standard (pd.DataFrame): The ground truth dataset.
        generated_dataset (pd.DataFrame): The dataset to evaluate.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    if gold_standard.shape != generated_dataset.shape:
        raise ValueError("Datasets must have the same shape for evaluation.")
    n_rows, n_cols = gold_standard.shape
    corrupted_coords_flattened = np.array(
        [coord for coords in corrupted_coords for coord in coords]
    )
    corrupted_coords_flattened = np.unique(corrupted_coords_flattened, axis=0)

    stats_per_column = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "column_names": list(gold_standard.columns),
        "stats": [],
    }

    for col in range(n_cols):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        column_corruption_coords = corrupted_coords_flattened[
            corrupted_coords_flattened[:, 1] == col
        ]
        stats_per_column["stats"].append(
            {
                "num_enties": n_rows,
                "column_name": gold_standard.columns[col],
            }
        )
        for row in range(n_rows):
            coord = np.array([row, col])
            is_corrupted = np.any(np.all(column_corruption_coords == coord, axis=1))
            gold_val = gold_standard.iloc[row, col]
            gen_val = generated_dataset.iloc[row, col]
            if is_corrupted and gold_val == gen_val:
                true_positive += 1
            elif not is_corrupted and gold_val != gen_val:
                false_positive += 1
            elif is_corrupted and gold_val != gen_val:
                false_negative += 1
            elif not is_corrupted and gold_val == gen_val:
                true_negative += 1
        stats_per_column["stats"][col].update(
            calculate_stats(
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                true_negative=true_negative,
            )
        )
    return stats_per_column
