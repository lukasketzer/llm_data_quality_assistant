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

    if rows_to_swap.size < 2:
        raise ValueError("You must provide at least two row indices to swap.")

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

    dataset.iloc[selected_row] = None

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
def shuffle_columns(dataset: pd.DataFrame, rows_to_shuffle: np.ndarray) -> pd.DataFrame:
    """
    Shuffle the columns for the specified rows in the dataset.
    """
    for row in rows_to_shuffle:
        dataset.iloc[row] = np.random.permutation(dataset.iloc[row])
    return dataset


def outlier(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    for row, col in cell_coordinates:
        if isinstance(dataset.iat[row, col], (int, float)):
            # Generate a random outlier value
            outlier_value = dataset.iat[row, col] * random.uniform(10, 100)
            dataset.iat[row, col] = outlier_value
        else:
            # Generate a random string as an outlier
            outlier_value = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=20))
            dataset.iat[row, col] = outlier_value

    return dataset


def null(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    for row, col in cell_coordinates:
        dataset.iat[row, col] = None
    return dataset


def adjacent_error(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    """
    Introduce small, plausible errors into a subset of the specified cell coordinates.
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
                if char in keyboard_flat:
                    # TODO: verticcal keys hinuzfü+ge
                    replacement_idx = keyboard_flat.index(char) + random.choice([-1, 1])
                    replacement_idx = max(
                        0, min(replacement_idx, len(keyboard_flat) - 1)
                    )
                    return (
                        value[:idx] + keyboard_flat[replacement_idx] + value[idx + 1 :]
                    )
                else:
                    # fallback: random letter
                    replacement = random.choice("abcdefghijklmnopqrstuvwxyz")
                    return value[:idx] + replacement + value[idx + 1 :]
            else:
                return value

        # Integers and floats: add or subtract a small random value
        # TODO: FIXME
        elif isinstance(value, (int, float)):
            return value + random.randint(-10, 10)
            print("here")
            if value == 0:
                return value + random.uniform(0.1, 1.0)
            noise = value * random.uniform(0.1, 1.0)
            if random.random() < 0.5:
                return value + noise
            else:
                return value - noise

        # Booleans: flip the value
        elif isinstance(value, bool):
            return not value

        # Dates and datetimes: add or subtract a small timedelta
        elif isinstance(value, (datetime.date, datetime.datetime, np.datetime64)):
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
    if cell_coordinates.size == 0:
        return dataset

    # FIX: Unpack row, col directly from cell_coordinates
    for row, col in cell_coordinates:
        value = dataset.iat[row, col]
        dataset.iat[row, col] = introduce_adjacent_error(value)
    return dataset


# TODO:
def contradictory_data(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray
) -> pd.DataFrame:
    return dataset


# TODO:
def incorrect_datatype(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray
) -> pd.DataFrame:
    return dataset


# TODO:
def inconsistent_format(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray
) -> pd.DataFrame:
    return dataset


def calculated_corruption_noise(
    dataset_dimensions: tuple[int, int],
    row_corruption_type: list[RowCorruptionTypes],
    cell_corruption_type: list[CellCorruptionTypes],
    severity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the noise to be added to the dataset based on the severity.
    """

    percentage_rows = (
        len(row_corruption_type)
        / (len(row_corruption_type) + len(cell_corruption_type))
        if (len(row_corruption_type) + len(cell_corruption_type)) > 0
        else 0
    )
    amount_rows_to_corrupt = math.ceil(
        dataset_dimensions[0] * severity * percentage_rows
    )
    percentage_cells = 1 - percentage_rows
    amount_cells_to_corrupt = math.ceil(
        dataset_dimensions[0] * dataset_dimensions[1] * severity * percentage_cells
    )

    # Rows
    if amount_rows_to_corrupt > 0:
        row_indices = np.random.choice(
            dataset_dimensions[0], amount_rows_to_corrupt, replace=False
        )
        row_coordinates = np.array(
            [(row, col) for row in row_indices for col in range(dataset_dimensions[1])]
        )
    else:
        row_indices = np.array([], dtype=int)
        row_coordinates = np.empty((0, 2), dtype=int)

    # Cells (unique)
    n_cells = dataset_dimensions[0] * dataset_dimensions[1]
    possible_cell_records = np.array(
        [
            (row, col)
            for row in range(dataset_dimensions[0])
            if row not in row_indices
            for col in range(dataset_dimensions[1])
        ]
    )
    if amount_cells_to_corrupt > 0:
        indices = np.random.choice(
            possible_cell_records.shape[0], amount_cells_to_corrupt, replace=False
        )
        cell_coordinates = possible_cell_records[indices]

    else:
        cell_coordinates = np.empty((0, 2), dtype=int)

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
    if len(corruption_types) == 0 or row_indices.size == 0:
        return dataset

    splits: list[np.ndarray] = np.array_split(row_indices, len(corruption_types))
    for split, c in zip(splits, corruption_types):
        match c:
            case RowCorruptionTypes.SWAP_ROWS:
                dataset = swap_rows(dataset, rows_to_swap=split)
            case RowCorruptionTypes.DELETE_ROWS:
                dataset = delete_rows(dataset, rows_to_delete=split)
            case RowCorruptionTypes.DUPLICATE_ROWS:
                dataset = duplicate_rows(dataset, rows_to_duplicate=split)
            case RowCorruptionTypes.SHUFFLE_COLUMNS:
                dataset = shuffle_columns(
                    dataset, rows_to_shuffle=split
                )  # severity not used
            case _:
                raise ValueError(f"Unknown corruption type: {c}")
    return dataset


def apply_cell_corruptions(
    corruption_types: list[CellCorruptionTypes],
    cell_coordinates: np.ndarray,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the specified cell corruption types to the dataset using the provided cell coordinates.
    Each corruption type is applied to an even split of the cell coordinates.
    """
    if (len(corruption_types) == 0) or (cell_coordinates.size == 0):
        return dataset

    splits: list[np.ndarray] = np.array_split(cell_coordinates, len(corruption_types))
    for split, c in zip(splits, corruption_types):
        match c:
            case CellCorruptionTypes.ADJACENT_ERROR:
                dataset = adjacent_error(dataset, split)
            case CellCorruptionTypes.OUTLIER:
                dataset = outlier(dataset, split)
            case CellCorruptionTypes.NULL:
                dataset = null(dataset, split)
            case CellCorruptionTypes.CONTRADICTORY_DATA:
                dataset = contradictory_data(dataset, split)
            case CellCorruptionTypes.INCORRECT_DATATYPE:
                dataset = incorrect_datatype(dataset, split)
            case CellCorruptionTypes.INCONSISTENT_FORMAT:
                dataset = inconsistent_format(dataset, split)
            case _:
                raise ValueError(f"Unknown cell corruption type: {c}")
    return dataset


def corrupt_dataset(
    gold_standard: pd.DataFrame,
    row_corruption_type: list[RowCorruptionTypes],
    cell_corruption_type: list[CellCorruptionTypes],
    severity: float = 0.1,  # Severity of corruption (0.0 to 1.0)
    output_size: int = 5,
) -> tuple[list[pd.DataFrame], list[np.ndarray]]:
    """
    Apply a corruption type to the dataset with a given severity.
    """

    corrupt_datasets = []
    corrupted_coords = []
    for d in range(output_size):
        dataset = gold_standard.copy()
        row_indices, cell_coordinates, merged_coordinates = calculated_corruption_noise(
            dataset_dimensions=dataset.shape,
            row_corruption_type=row_corruption_type,
            cell_corruption_type=cell_corruption_type,
            severity=severity,
        )
        print(f"merged coords: {merged_coordinates.shape}")
        corrupted_coords.append(merged_coordinates)

        dataset = apply_row_corruptions(
            corruption_types=row_corruption_type,
            row_indices=row_indices,
            dataset=dataset,
        )
        dataset = apply_cell_corruptions(
            corruption_types=cell_corruption_type,
            cell_coordinates=cell_coordinates,
            dataset=dataset,
        )

        corrupt_datasets.append(dataset)
    return corrupt_datasets, corrupted_coords


def analyze_dataset(
    gold_standard: pd.DataFrame,
    corrupted_dataset: pd.DataFrame,
) -> float:
    """
    Analyze a dataset for a given corruption type and severity.
    """

    print(gold_standard)
    print(corrupted_dataset)

    n_total: int = gold_standard.shape[0] * gold_standard.shape[1]
    n_corrupted: int = (
        (gold_standard != corrupted_dataset).sum().sum()
    )  # Count the number of corrupted cells
    return n_corrupted / n_total


if __name__ == "__main__":
    dataset = pd.read_csv("datasets/selfwritte_dataset/dataset.csv")
    dataset = pd.read_csv("datasets/public_dataset/wine.data")
    print(dataset)
    print(dataset.dtypes)
    # Test 1: Only cell corruption (NULL)
    corrupted_datasets, corrupted_coords = corrupt_dataset(
        gold_standard=dataset,
        row_corruption_type=[RowCorruptionTypes.DELETE_ROWS],
        cell_corruption_type=[],
        severity=0.14,
        output_size=1,
    )
    for i, corrupted_dataset in enumerate(corrupted_datasets):
        print(f"Test 1 - Corrupted Dataset {i + 1}")
        print(
            f"Analysis: {analyze_dataset(dataset, corrupted_dataset) * 100:.2f}% of the dataset is corrupted."
        )

    print(corrupted_datasets[0])
