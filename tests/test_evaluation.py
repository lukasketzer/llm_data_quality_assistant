import sys
import os

# Add the project root to sys.path for pytest run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from analysis import evaluation


def test_precision():
    assert evaluation.precision(5, 0) == 1.0
    assert evaluation.precision(0, 5) == 0.0
    assert evaluation.precision(3, 2) == 0.6
    assert evaluation.precision(0, 0) == 0.0


def test_recall():
    assert evaluation.recall(5, 0) == 1.0
    assert evaluation.recall(0, 5) == 0.0
    assert evaluation.recall(3, 2) == 0.6
    assert evaluation.recall(0, 0) == 0.0


def test_f1_score():
    assert evaluation.f1_score(5, 0, 0) == 1.0
    assert evaluation.f1_score(0, 5, 5) == 0.0
    assert evaluation.f1_score(2, 1, 1) == 0.6666666666666666
    assert evaluation.f1_score(0, 0, 0) == 0.0


def test_accuracy():
    assert evaluation.accuracy(5, 5, 0, 0) == 1.0
    assert evaluation.accuracy(0, 0, 5, 5) == 0.0
    assert evaluation.accuracy(2, 3, 1, 4) == 0.5
    assert evaluation.accuracy(0, 0, 0, 0) == 0.0


def test_false_positive_rate():
    assert evaluation.false_positive_rate(0, 5) == 0.0
    assert evaluation.false_positive_rate(5, 0) == 1.0
    assert evaluation.false_positive_rate(2, 2) == 0.5
    assert evaluation.false_positive_rate(0, 0) == 0.0


def test_false_negative_rate():
    assert evaluation.false_negative_rate(0, 5) == 0.0
    assert evaluation.false_negative_rate(5, 0) == 1.0
    assert evaluation.false_negative_rate(2, 2) == 0.5
    assert evaluation.false_negative_rate(0, 0) == 0.0


def test_calculate_stats():
    stats = evaluation.calculate_stats(2, 1, 1, 6)
    assert stats["precision"] == 2 / 3
    assert stats["recall"] == 2 / 3
    assert stats["f1_score"] == 2 / 3
    assert stats["accuracy"] == 8 / 10
    assert stats["false_positive_rate"] == 1 / 7
    assert stats["false_negative_rate"] == 1 / 3


def test_evaluate_dataset_micro():
    gold = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
    cleaned = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 0]})
    corrupted = pd.DataFrame({"id": [1, 2], "a": [1, 0], "b": [0, 4]})
    stats = evaluation.evaluate_dataset_micro(
        gold, cleaned, corrupted, primary_key="id"
    )
    assert stats["true_positive"] >= 0
    assert stats["false_positive"] >= 0
    assert stats["false_negative"] >= 0
    assert stats["true_negative"] >= 0
    assert "precision" in stats
    assert "recall" in stats
    assert "f1_score" in stats
    assert "accuracy" in stats
    assert "false_positive_rate" in stats
    assert "false_negative_rate" in stats


def test_evaluate_dataset_macro():
    gold = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
    cleaned = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 0]})
    corrupted = pd.DataFrame({"id": [1, 2], "a": [1, 0], "b": [0, 4]})
    stats = evaluation.evaluate_dataset_macro(
        gold, cleaned, corrupted, primary_key="id"
    )
    assert "stats" in stats
    # Only non-primary-key columns are evaluated
    assert len(stats["stats"]) == 2
    for col_stats in stats["stats"]:
        assert "precision" in col_stats
        assert "recall" in col_stats
        assert "f1_score" in col_stats
        assert "accuracy" in col_stats
        assert "false_positive_rate" in col_stats
        assert "false_negative_rate" in col_stats


def test_evaluate_dataset_micro_shape_error():
    gold = pd.DataFrame({"id": [1, 2], "a": [1, 2]})
    cleaned = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
    corrupted = pd.DataFrame({"id": [1, 2], "a": [1, 0], "b": [0, 4]})
    with pytest.raises(ValueError):
        evaluation.evaluate_dataset_micro(gold, cleaned, corrupted, primary_key="id")


def test_evaluate_dataset_macro_shape_error():
    gold = pd.DataFrame({"id": [1, 2], "a": [1, 2]})
    cleaned = pd.DataFrame({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
    corrupted = pd.DataFrame({"id": [1, 2], "a": [1, 0], "b": [0, 4]})
    with pytest.raises(ValueError):
        evaluation.evaluate_dataset_macro(gold, cleaned, corrupted, primary_key="id")


def test_evaluate_empty_dataframes():
    gold = pd.DataFrame(columns=["id", "a", "b"])
    cleaned = pd.DataFrame(columns=["id", "a", "b"])
    corrupted = pd.DataFrame(columns=["id", "a", "b"])
    stats_micro = evaluation.evaluate_dataset_micro(
        gold, cleaned, corrupted, primary_key="id"
    )
    stats_macro = evaluation.evaluate_dataset_macro(
        gold, cleaned, corrupted, primary_key="id"
    )
    assert stats_micro["num_rows"] == 0
    assert stats_micro["num_columns"] == 3
    assert stats_macro["num_rows"] == 0
    assert stats_macro["num_columns"] == 3


def test_evaluate_nan_dataframes():
    gold = pd.DataFrame({"id": [1, 2], "a": [float("nan"), float("nan")]})
    cleaned = pd.DataFrame({"id": [1, 2], "a": [float("nan"), float("nan")]})
    corrupted = pd.DataFrame({"id": [1, 2], "a": [float("nan"), float("nan")]})
    stats = evaluation.evaluate_dataset_micro(
        gold, cleaned, corrupted, primary_key="id"
    )
    assert "precision" in stats
    assert "recall" in stats
    assert "f1_score" in stats
    assert "accuracy" in stats
    assert "false_positive_rate" in stats
    assert "false_negative_rate" in stats


def test_calculate_stats_types():
    stats = evaluation.calculate_stats(1, 2, 3, 4)
    assert isinstance(stats["precision"], float)
    assert isinstance(stats["recall"], float)
    assert isinstance(stats["f1_score"], float)
    assert isinstance(stats["accuracy"], float)
    assert isinstance(stats["false_positive_rate"], float)
    assert isinstance(stats["false_negative_rate"], float)
