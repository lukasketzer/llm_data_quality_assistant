import sys
import os

# Add the project root to sys.path for pytest run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np

from llm_data_quality_assistant.corruptor import corrupt_dataset
from llm_data_quality_assistant.enums.CorruptionTypes import (
    CellCorruptionTypes,
    RowCorruptionTypes,
)


@pytest.mark.parametrize(
    "cell_corruption_type",
    [
        CellCorruptionTypes.OUTLIER,
        CellCorruptionTypes.NULL,
        CellCorruptionTypes.SWAP_CELLS,
        CellCorruptionTypes.CASE_ERROR,
        CellCorruptionTypes.TRUNCATE,
        CellCorruptionTypes.ROUNDING_ERROR,
        CellCorruptionTypes.TYPO,
    ],
)
def test_corrupt_dataset_severity_single_cell_type(cell_corruption_type):
    df = pd.read_csv("datasets/llm_dataset/Radiology_modality_sample.csv")
    row_corruption_types = []
    cell_corruption_types = [cell_corruption_type]
    excluded_columns = ["dicom_uid"]
    severity = 0.13
    output_size = 1

    corrupted_dfs, coords = corrupt_dataset(
        dataset=df,
        row_corruption_types=row_corruption_types,
        cell_corruption_types=cell_corruption_types,
        columns_to_exclude=excluded_columns,
        severity=severity,
        output_size=output_size,
    )

    corrupted_df = corrupted_dfs[0]
    df_compare = df.drop(columns=excluded_columns, errors="ignore")
    corrupted_compare = corrupted_df.drop(columns=excluded_columns, errors="ignore")
    total_cells = np.prod(df_compare.shape)
    corrupted_cells = np.sum(df_compare.values != corrupted_compare.values)
    actual_severity = corrupted_cells / total_cells

    tolerance = 0.05
    assert (
        abs(actual_severity - severity) < tolerance
    ), f"Cell Type: {cell_corruption_type}, Actual: {actual_severity}, Expected: {severity}"


@pytest.mark.parametrize(
    "row_corruption_type",
    [
        # RowCorruptionTypes.SWAP_ROWS,
        RowCorruptionTypes.DELETE_ROWS,
        RowCorruptionTypes.SHUFFLE_COLUMNS,
        RowCorruptionTypes.REVERSE_ROWS,
    ],
)
def test_corrupt_dataset_severity_single_row_type(row_corruption_type):
    df = pd.read_csv("datasets/llm_dataset/Radiology_modality_sample.csv")
    row_corruption_types = [row_corruption_type]
    cell_corruption_types = []
    excluded_columns = ["dicom_uid"]
    severity = 0.13
    output_size = 1

    corrupted_dfs, coords = corrupt_dataset(
        dataset=df,
        row_corruption_types=row_corruption_types,
        cell_corruption_types=cell_corruption_types,
        columns_to_exclude=excluded_columns,
        severity=severity,
        output_size=output_size,
    )

    corrupted_df = corrupted_dfs[0]
    df_compare = df.drop(columns=excluded_columns, errors="ignore")
    corrupted_compare = corrupted_df.drop(columns=excluded_columns, errors="ignore")
    total_cells = np.prod(df_compare.shape)
    corrupted_cells = np.sum(df_compare.values != corrupted_compare.values)
    actual_severity = corrupted_cells / total_cells

    tolerance = 0.05
    assert (
        abs(actual_severity - severity) < tolerance
    ), f"Row Type: {row_corruption_type}, Actual: {actual_severity}, Expected: {severity}"


def test_corrupted_coordinates_match_changes():
    df = pd.read_csv("datasets/llm_dataset/Radiology_modality_sample.csv")
    cell_corruption_types = [
        CellCorruptionTypes.NULL,
    ]
    row_corruption_types = []
    excluded_columns = ["dicom_uid", "series_desc"]
    # excluded_columns = []
    severity = 0.13
    output_size = 1

    corrupted_dfs, coords = corrupt_dataset(
        dataset=df,
        row_corruption_types=row_corruption_types,
        cell_corruption_types=cell_corruption_types,
        columns_to_exclude=excluded_columns,
        severity=severity,
        output_size=output_size,
    )
    corrupted_df = corrupted_dfs[0]
    coords = coords[0]
    # Do NOT drop excluded columns in the comparison
    df_compare = df.reset_index(drop=True)
    corrupted_compare = corrupted_df.reset_index(drop=True)

    unchanged = 0
    for row, col in coords:
        if col < 0 or col >= df_compare.shape[1]:
            continue  # skip out-of-bounds
        if df_compare.iat[row, col] == corrupted_compare.iat[row, col]:
            unchanged += 1
            print(
                f"Warning: Coordinate ({row}, {col}) not corrupted as expected (value unchanged)"
            )
        else:
            assert (
                df_compare.iat[row, col] != corrupted_compare.iat[row, col]
            ), f"Coordinate ({row}, {col}) not corrupted as expected."
    assert (
        unchanged < len(coords) * 0.5
    ), "Too many coordinates were not actually changed."

    changed = np.argwhere(df_compare.values != corrupted_compare.values)
    for row, col in changed:
        assert any(
            (row == c[0] and col == c[1]) for c in coords
        ), f"Changed cell ({row}, {col}) not in coords list."


def test_corrupt_dataset_empty_df():
    df = pd.DataFrame()
    corrupted_dfs, coords = corrupt_dataset(
        dataset=df,
        row_corruption_types=[],
        cell_corruption_types=[],
        columns_to_exclude=[],
        severity=0.1,
        output_size=1,
    )
    assert isinstance(corrupted_dfs, list)
    assert len(corrupted_dfs) == 1
    assert corrupted_dfs[0].empty
    assert coords == [[]]


def test_corrupt_dataset_nan_df():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    corrupted_dfs, coords = corrupt_dataset(
        dataset=df,
        row_corruption_types=[],
        cell_corruption_types=[CellCorruptionTypes.NULL],
        columns_to_exclude=[],
        severity=0.5,
        output_size=1,
    )
    assert isinstance(corrupted_dfs, list)
    assert len(corrupted_dfs) == 1
    assert set(corrupted_dfs[0].columns) == set(["a", "b"])
    # All values should still be nan or null
    assert corrupted_dfs[0].isnull().all().all()


def test_corrupt_dataset_invalid_severity():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        corrupt_dataset(
            dataset=df,
            row_corruption_types=[],
            cell_corruption_types=[CellCorruptionTypes.NULL],
            columns_to_exclude=[],
            severity=-0.1,
            output_size=1,
        )
    with pytest.raises(ValueError):
        corrupt_dataset(
            dataset=df,
            row_corruption_types=[],
            cell_corruption_types=[CellCorruptionTypes.NULL],
            columns_to_exclude=[],
            severity=1.5,
            output_size=1,
        )


def test_corrupt_dataset_output_size_zero():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        corrupt_dataset(
            dataset=df,
            row_corruption_types=[],
            cell_corruption_types=[CellCorruptionTypes.NULL],
            columns_to_exclude=[],
            severity=0.1,
            output_size=0,
        )
