import numpy as np
import pandas as pd
from corruptor import corrupt_dataset, RowCorruptionTypes, CellCorruptionTypes


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


def evaluate_datsets(
    gold_standard: pd.DataFrame,
    generated_dataset: pd.DataFrame,
    corrupted_coords: np.ndarray,
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

    # Calculate basic statistics
    stats = {
        "num_rows": n_rows,
        "num_columns": n_cols,
        "column_names": list(generated_dataset.columns),
    }

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for row in range(n_rows):
        for col in range(n_cols):
            coord = np.array([row, col])
            is_corrupted = np.any(np.all(corrupted_coords == coord, axis=1))
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
    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    f1 = f1_score(true_positive, false_positive, false_negative)
    acc = accuracy(true_positive, true_negative, false_positive, false_negative)
    fpr = false_positive_rate(false_positive, true_negative)
    fnr = false_negative_rate(false_negative, true_positive)
    stats.update(
        {
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
    )

    return stats


if __name__ == "__main__":
    # dataset = pd.read_csv("datasets/selfwritte_dataset/dataset.csv")
    dataset = pd.read_csv("datasets/public_dataset/wine.data")
    corrupted_datasets, corrupted_coords = corrupt_dataset(
        gold_standard=dataset,
        row_corruption_type=[],
        cell_corruption_type=[CellCorruptionTypes.NULL],
        severity=0.13,
        output_size=2,
    )
    print(type(corrupted_coords[0][0]))
    print(corrupted_coords[0][0] in corrupted_coords[0])

    stats = evaluate_datsets(dataset, corrupted_datasets[0], corrupted_coords[0])
    print(stats)
