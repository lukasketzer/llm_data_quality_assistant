import sys
import os

# Add the project root to sys.path for pytest run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from llm_data_quality_assistant.pipeline import Pipeline


def test_standardize_datasets_row_and_column_order():
    # Create shuffled DataFrames with same data
    data = {"id": [2, 1, 3], "A": [20, 10, 30], "B": [200, 100, 300]}
    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame({"B": [100, 200, 300], "A": [10, 20, 30], "id": [1, 2, 3]})[
        ["id", "A", "B"]
    ]  # reorder columns to match df1
    df2 = df2.sample(frac=1).reset_index(drop=True)  # shuffle rows
    # Intentionally shuffle columns for df3
    df3 = pd.DataFrame({"A": [30, 10, 20], "id": [3, 1, 2], "B": [300, 100, 200]})[
        ["id", "A", "B"]
    ]
    df3 = df3.sample(frac=1).reset_index(drop=True)

    # Standardize
    result = Pipeline.standardize_datasets(primary_key="id", df1=df1, df2=df2, df3=df3)
    # All should have same row order and column order
    for name, df in result.items():
        assert list(df.columns) == ["id", "A", "B"]
        assert list(df["id"]) == [1, 2, 3]
    # All values should match after sorting
    assert result["df1"].equals(result["df2"])
    assert result["df1"].equals(result["df3"])


def test_standardize_datasets_column_order_only():
    # Test that columns are aligned even if order is different
    data = {"id": [1, 2], "A": [10, 20], "B": [100, 200]}
    df1 = pd.DataFrame(data)
    df2 = pd.DataFrame({"B": [100, 200], "A": [10, 20], "id": [1, 2]})
    result = Pipeline.standardize_datasets(primary_key="id", df1=df1, df2=df2)
    for df in result.values():
        assert list(df.columns) == ["id", "A", "B"]
        assert list(df["id"]) == [1, 2]
    assert result["df1"].equals(result["df2"])


def test_standardize_datasets_multiple_shuffled_copies():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.abspath(
        os.path.join(
            base_dir, "../datasets/self_generated_dataset/Radiology_modality_sample.csv"
        )
    )
    df = pd.read_csv(data_path).head(20)
    dfs = {}
    for i in range(5):
        shuffled = df.sample(frac=1).reset_index(drop=True)
        shuffled = shuffled[shuffled.columns[::-1]]  # reverse columns for extra test
        dfs[f"df{i}"] = shuffled
    result = Pipeline.standardize_datasets(primary_key="dicom_uid", **dfs)
    # All standardized DataFrames should have the same column order and row order
    columns_list = [list(df.columns) for df in result.values()]
    assert all(cols == columns_list[0] for cols in columns_list)
    id_list = [list(df["dicom_uid"]) for df in result.values()]
    assert all(ids == id_list[0] for ids in id_list)
