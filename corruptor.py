from faker import Faker
import numpy
import random
import numpy as np
from enum import Enum
import pandas as pd
import math


class RowCorruptionTypes(Enum):
    SWAP_ROWS = "swap_rows"
    DELETE_ROWS = "delete_rows"
    DUPLICATE_ROWS = "duplicate_rows"
    SHUFFLE_COLUMNS = "shuffle_columns"


class CellCorruptionTypes(Enum):
    MISSING_VALUES = "missing_values"
    OUTLIER = "outlier"
    NULL = "null"
    CONTRADICTORY_DATA = "contradictory_data"
    INCORRECT_DATATYPE = "incorrect_datatype"
    INCONSISTENT_FORMAT = "inconsistent_format"
    ADJACENT_ERROR = "adjacent_error"


# Franzi
def swap_rows(dataset: pd.DataFrame, rows_to_swap: np.ndarray) -> pd.DataFrame:
    n_rows = dataset.shape[0]

    if rows_to_swap.size == 0:
        raise ValueError("You must provide a non-empty list of row indices to swap.")

    if np.any(rows_to_swap >= n_rows):
        raise IndexError("Row indices out of bounds.")

    # Generate a derangement (no index stays in place)
    while True:
        perm = np.random.permutation(rows_to_swap)
        if not np.any(perm == rows_to_swap):
            dataset.iloc[rows_to_swap] = dataset.iloc[perm].values
            break

    return dataset


# Franzi
def delete_rows(dataset: pd.DataFrame, rows_to_delete: np.ndarray) -> pd.DataFrame:
    n_rows = dataset.shape[0]

    if rows_to_delete.size == 0:
        raise ValueError("You must provide a non-empty list of row indices to delete.")

    # Pick one row index from the provided list
    selected_row = random.choice(rows_to_delete)

    if selected_row >= n_rows:
        raise IndexError(
            f"Row index {selected_row} is out of bounds for dataset with {n_rows} rows."
        )

    # Drop the selected row
    dataset = dataset.drop(index=selected_row).reset_index(drop=True)

    return dataset


# Franzi
def duplicate_rows(
    dataset: pd.DataFrame, rows_to_duplicate: np.ndarray
) -> pd.DataFrame:
    n_rows = dataset.shape[0]

    if rows_to_duplicate.size == 0:
        raise ValueError(
            "You must provide a non-empty list of row indices to duplicate."
        )

    # Pick one row index from the provided list
    selected_row = random.choice(rows_to_duplicate)

    if selected_row >= n_rows:
        raise IndexError(
            f"Row index {selected_row} is out of bounds for dataset with {n_rows} rows."
        )

    # Duplicate the selected row
    row_to_duplicate = dataset.iloc[[selected_row]]
    dataset = pd.concat(
        [
            dataset.iloc[: selected_row + 1],
            row_to_duplicate,
            dataset.iloc[selected_row + 1 :],
        ],
        ignore_index=True,
    )

    return dataset


# Franzi
def shuffle_columns(
    dataset: pd.DataFrame, rows_to_shuffle: list[int], severity: float
) -> pd.DataFrame:
    """
    Shuffle the columns for the specified rows in the dataset.
    """
    for row in rows_to_shuffle:
        dataset.iloc[row] = np.random.permutation(dataset.iloc[row])
    return dataset


# Franzi
def missing_values(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray, severity: float
) -> pd.DataFrame:
    """
    Set a subset of the specified cell coordinates to NaN, based on severity.
    cell_coordinates: np.ndarray of shape (N, 2), where each row is (row_idx, col_idx)
    severity: float between 0 and 1, fraction of cell_coordinates to corrupt
    """
    n_total = len(cell_coordinates)
    n_corrupt = max(1, int(severity * n_total)) if n_total > 0 else 0
    if n_corrupt == 0:
        return dataset
    # Randomly select indices to corrupt
    selected_indices = np.random.choice(n_total, n_corrupt, replace=False)
    for idx in selected_indices:
        row, col = cell_coordinates[idx]
        dataset.iat[row, col] = np.nan
    return dataset


def outlier(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    for row, col in cell_coordinates:
        if isinstance(dataset.iat[row, col], (int, float)):
            # Generate a random outlier value
            outlier_value = dataset.iat[row, col] * random.uniform(10, 100)
            dataset.iat[row, col] = outlier_value
        if isinstance(dataset.iat[row, col], str):
            # Generate a random string as an outlier
            outlier_value = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=20))
            dataset.iat[row, col] = outlier_value

    return dataset


def null(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    for row, col in cell_coordinates:
        dataset.iat[row, col] = None
    return dataset


def adjacent_error(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray, severity: float
) -> pd.DataFrame:
    """
    Introduce small, plausible errors into a subset of the specified cell coordinates, based on severity.
    Handles strings, numbers, booleans, dates, and other common types.
    """

    import datetime

    def introduce_adjacent_error(value):
        # Strings: swap, delete, or replace a character with a similar one
        if isinstance(value, str):
            if len(value) < 1:
                return value
            error_type = random.choice(["swap", "delete", "replace"])
            idx = random.randint(0, len(value) - 1)
            if error_type == "swap" and len(value) > 1 and idx < len(value) - 1:
                s_list = list(value)
                s_list[idx], s_list[idx + 1] = s_list[idx + 1], s_list[idx]
                return "".join(s_list)
            elif error_type == "delete" and len(value) > 1:
                return value[:idx] + value[idx + 1 :]
            elif error_type == "replace":
                # TODO: rework

                keyboard_flat = [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "0",
                    "q",
                    "w",
                    "e",
                    "r",
                    "t",
                    "y",
                    "u",
                    "i",
                    "o",
                    "p",
                    "a",
                    "s",
                    "d",
                    "f",
                    "g",
                    "h",
                    "j",
                    "k",
                    "l",
                    "z",
                    "x",
                    "c",
                    "v",
                    "b",
                    "n",
                    "m",
                ]
                char = value[idx]

                replacement = keyboard_flat.index(char) + random.choice([-1, 1])
                return value[:idx] + keyboard_flat[replacement] + value[idx + 1 :]
            else:
                return value

        # Integers and floats: add or subtract a small random value
        elif isinstance(value, (int, float)):
            if value == 0:
                return value + random.uniform(0.1, 1.0)
            noise = value * random.uniform(0.01, 0.1)
            if random.random() < 0.5:
                return value + noise
            else:
                return value - noise

        # Booleans: flip the value
        elif isinstance(value, bool):
            return not value

        # Dates and datetimes: add or subtract a small timedelta
        elif isinstance(value, (datetime.date, datetime.datetime, np.datetime64)):
            # Convert np.datetime64 to datetime for manipulation
            if isinstance(value, np.datetime64):
                value = pd.to_datetime(value)
            delta_days = random.randint(1, 5)
            if random.random() < 0.5:
                return value + datetime.timedelta(days=delta_days)
            else:
                return value - datetime.timedelta(days=delta_days)

        # Timedelta: add or subtract a small amount
        elif isinstance(value, datetime.timedelta):
            delta = datetime.timedelta(days=random.randint(1, 3))
            if random.random() < 0.5:
                return value + delta
            else:
                return value - delta

        # Fallback: return value unchanged
        else:
            return value

    n_total = len(cell_coordinates)
    n_corrupt = max(1, int(severity * n_total)) if n_total > 0 else 0
    if n_corrupt == 0:
        return dataset

    selected_indices = np.random.choice(n_total, n_corrupt, replace=False)
    for idx in selected_indices:
        row, col = cell_coordinates[idx]
        value = dataset.iat[row, col]
        dataset.iat[row, col] = introduce_adjacent_error(value)
    return dataset


# TODO:
def contradictory_data(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray, severity: float
) -> pd.DataFrame:
    return pd.DataFrame()


# TODO:
def incorrect_datatype(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray, severity: float
) -> pd.DataFrame:
    return pd.DataFrame()


# TODO:
def inconsistent_format(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray, severity: float
) -> pd.DataFrame:
    return pd.DataFrame()


def calculated_corruption_noise(
    dataset_dimensions: tuple[int, int], severity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the noise to be added to the dataset based on the severity.
    """

    percentage_rows = 0.2
    amount_rows_to_corrupt = math.floor(
        dataset_dimensions[0] * severity * percentage_rows
    )
    percentage_cells = 1 - percentage_rows
    amount_cells_to_corrupt = math.ceil(
        dataset_dimensions[0] * dataset_dimensions[1] * severity * percentage_cells
    )

    # Rows
    if amount_rows_to_corrupt > 0:
        row_indices = np.random.randint(
            0, dataset_dimensions[0], amount_rows_to_corrupt
        )
        row_coordinates = np.array(
            [(row, col) for row in row_indices for col in range(dataset_dimensions[1])]
        )
    else:
        row_indices = np.array([], dtype=int)
        row_coordinates = np.empty((0, 2), dtype=int)

    # Cells
    if amount_cells_to_corrupt > 0:
        x_coords = np.random.randint(0, dataset_dimensions[0], amount_cells_to_corrupt)
        y_coords = np.random.randint(0, dataset_dimensions[1], amount_cells_to_corrupt)
        cell_coordinates = np.column_stack((x_coords, y_coords))
    else:
        cell_coordinates = np.empty((0, 2), dtype=int)

    # Combine into (x, y) pairs
    cell_coordinates = np.column_stack((x_coords, y_coords))

    # Merge row_coordinates and cell_coordinates
    merged_coordinates = np.vstack((row_coordinates, cell_coordinates))

    return (
        row_indices,  # np.ndarray of row indices
        cell_coordinates,  # np.ndarray of individual cell coordinates
        merged_coordinates,  # np.ndarray of all merged coordinates
    )


def apply_row_corruptions(
    corruption_types: list[RowCorruptionTypes],
    row_indices: np.ndarray,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the specified row corruption types to the dataset using the provided row indices.
    """
    splits: list[np.ndarray] = np.array_split(row_indices, len(corruption_types))
    for split, c in zip(splits, corruption_types):
        match c:
            case RowCorruptionTypes.SWAP_ROWS:
                dataset = swap_rows(dataset, rows_to_swap=split)
            case RowCorruptionTypes.DELETE_ROWS:
                dataset = delete_rows(dataset, rows_to_delete=split)
            case RowCorruptionTypes.DUPLICATE_ROWS:
                dataset = duplicate_rows(dataset, rows_to_duplicate=split.tolist())
            case RowCorruptionTypes.SHUFFLE_COLUMNS:
                dataset = shuffle_columns(
                    dataset, rows_to_shuffle=split.tolist(), severity=0
                )  # severity not used
            case _:
                raise ValueError(f"Unknown corruption type: {c}")
    return dataset


def apply_cell_corruptions(
    corruption_types: list[CellCorruptionTypes],
    cell_coordinates: np.ndarray,
    dataset: pd.DataFrame,
    severity: float,
) -> pd.DataFrame:
    """
    Apply the specified cell corruption types to the dataset using the provided cell coordinates.
    Each corruption type is applied to an even split of the cell coordinates.
    """
    splits: list[np.ndarray] = np.array_split(cell_coordinates, len(corruption_types))
    for split, c in zip(splits, corruption_types):
        match c:
            case CellCorruptionTypes.MISSING_VALUES:
                dataset = missing_values(dataset, split, severity)
            case CellCorruptionTypes.ADJACENT_ERROR:
                dataset = adjacent_error(dataset, split, severity)
            case CellCorruptionTypes.OUTLIER:
                dataset = outlier(dataset, split)
            case CellCorruptionTypes.NULL:
                dataset = null(dataset, split)
            case CellCorruptionTypes.CONTRADICTORY_DATA:
                dataset = contradictory_data(dataset, split, severity)
            case CellCorruptionTypes.INCORRECT_DATATYPE:
                dataset = incorrect_datatype(dataset, split, severity)
            case CellCorruptionTypes.INCONSISTENT_FORMAT:
                dataset = inconsistent_format(dataset, split, severity)
            case _:
                raise ValueError(f"Unknown cell corruption type: {c}")
    return dataset


def corrupt_dataset(
    gold_standard: pd.DataFrame,
    row_corruption_type: list[RowCorruptionTypes] = [RowCorruptionTypes.SWAP_ROWS],
    cell_corruption_type: list[CellCorruptionTypes] = [
        CellCorruptionTypes.MISSING_VALUES,
        CellCorruptionTypes.ADJACENT_ERROR,
    ],
    severity: float = 0.1,  # Severity of corruption (0.0 to 1.0)
    output_size: int = 5,
) -> list[pd.DataFrame]:
    """
    Apply a corruption type to the dataset with a given severity.
    """

    corrupt_datasets = []
    for d in range(output_size):
        dataset = gold_standard.copy()
        row_indices, cell_coordinates, merged_coordinates = calculated_corruption_noise(
            dataset_dimensions=dataset.shape, severity=severity
        )
        dataset = apply_row_corruptions(
            corruption_types=row_corruption_type,
            row_indices=row_indices,
            dataset=dataset,
        )
        dataset = apply_cell_corruptions(
            corruption_types=cell_corruption_type,
            cell_coordinates=cell_coordinates,
            dataset=dataset,
            severity=severity,
        )

        corrupt_datasets.append(dataset)
    return corrupt_datasets


def analyze_dataset(
    gold_standard: pd.DataFrame,
    corrupted_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze a dataset for a given corruption type and severity.
    """
    return pd.DataFrame()


if __name__ == "__main__":
    dataset = pd.read_csv("datasets/public_dataset/wine.data", header=None)
    print(dataset)
    row_indices, cell_coordinates, merged_coordinates = calculated_corruption_noise(
        dataset_dimensions=dataset.shape, severity=0.15
    )
    print(f"row_indices: {row_indices}")
    print(f"Cell coordinates: {cell_coordinates}")
    print(f"Merged coordinates: {merged_coordinates}")
    print(
        f"corrution percentage: {len(merged_coordinates) / (dataset.shape[0] * dataset.shape[1])}"
    )
    pass
