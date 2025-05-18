from faker import Faker
import numpy
import random
import numpy as np
from enum import Enum
import pandas as pd
import math


class CorruptionTypes(Enum):
    """
    Enum for the different types of corruption that can be applied to a dataset.
    """

    SWAP_ROWS = "swap_rows"
    DELETE_ROWS = "delete_rows"
    MISSING_VALUES = "missing_values"
    DUPLICATE_ROWS = "duplicate_rows"
    TYPO = "typo"
    OUTLIER = "outlier"
    NULL = "null"
    CONTRADICTORY_DATA = "contradictory_data"
    INCORRECT_DATATYPE = "incorrect_datatype"
    INCONSISTENT_FORMAT = "inconsistent_format"
    SHUFFLE_COLUMNS = "shuffle_columns"


# Franzi
def swap_rows(dataset: pd.DataFrame, rows_to_swap: list[int] = None) -> pd.DataFrame:
    n_rows = dataset.shape[0]

    if not rows_to_swap:
        raise ValueError("You must provide a non-empty list of row indices to swap.")

    # Pick one row index from the provided list
    selected_row = random.choice(rows_to_swap)

    if selected_row >= n_rows:
        raise IndexError(f"Row index {selected_row} is out of bounds for dataset with {n_rows} rows.")

    # Pick a different random row index to swap with
    other_row = random.choice([i for i in range(n_rows) if i != selected_row])

    # Swap the rows
    temp = dataset.iloc[selected_row].copy()
    dataset.iloc[selected_row] = dataset.iloc[other_row]
    dataset.iloc[other_row] = temp

    return dataset


# Franzi
def delete_rows(dataset: pd.DataFrame, rows_to_delete: list[int] = None) -> pd.DataFrame:
    n_rows = dataset.shape[0]

    if not rows_to_delete:
        raise ValueError("You must provide a non-empty list of row indices to delete.")

    # Pick one row index from the provided list
    selected_row = random.choice(rows_to_delete)

    if selected_row >= n_rows:
        raise IndexError(f"Row index {selected_row} is out of bounds for dataset with {n_rows} rows.")

    # Drop the selected row
    dataset = dataset.drop(index=selected_row).reset_index(drop=True)

    return dataset


# Franzi
def missing_values(dataset: pd.DataFrame, cell_coordinates: np.ndarray, severity: float) -> pd.DataFrame:
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


# Franzi
def duplicate_rows(dataset: pd.DataFrame, rows_to_duplicate: list[int] = None) -> pd.DataFrame:
    n_rows = dataset.shape[0]

    if not rows_to_duplicate:
        raise ValueError("You must provide a non-empty list of row indices to duplicate.")

    # Pick one row index from the provided list
    selected_row = random.choice(rows_to_duplicate)

    if selected_row >= n_rows:
        raise IndexError(f"Row index {selected_row} is out of bounds for dataset with {n_rows} rows.")

    # Duplicate the selected row
    row_to_duplicate = dataset.iloc[[selected_row]]
    dataset = pd.concat([dataset.iloc[:selected_row + 1], row_to_duplicate, dataset.iloc[selected_row + 1:]], ignore_index=True)

    return dataset


# Franzi
def typo(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


# Franzi
def outlier(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


def null(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


def contradictory_data(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


def incorrect_datatype(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


def inconsistent_format(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


def shuffle_columns(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


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


def corrupt_dataset(
    gold_standard: pd.DataFrame,
    corruption_type: list[CorruptionTypes] = [CorruptionTypes.SWAP_ROWS],
    severity: float = 0.1,  # Severity of corruption (0.0 to 1.0)
    output_size: int = 5,
) -> pd.DataFrame:
    """
    Apply a corruption type to the dataset with a given severity.
    """

    corrupt_datasets = []
    for d in range(output_size):
        dataset = gold_standard.copy()

        for corruption in corruption_type:
            match corruption:
                case CorruptionTypes.SWAP_ROWS:
                    dataset = swap_rows(dataset, severity)
                case CorruptionTypes.DELETE_ROWS:
                    dataset = delete_rows(dataset, severity)
                case CorruptionTypes.MISSING_VALUES:
                    dataset = missing_values(dataset, severity)
                case CorruptionTypes.DUPLICATE_ROWS:
                    dataset = duplicate_rows(dataset, severity)
                case CorruptionTypes.TYPO:
                    dataset = typo(dataset, severity)
                case CorruptionTypes.OUTLIER:
                    dataset = outlier(dataset, severity)
                case CorruptionTypes.NULL:
                    dataset = null(dataset, severity)
                case CorruptionTypes.CONTRADICTORY_DATA:
                    dataset = contradictory_data(dataset, severity)
                case CorruptionTypes.INCORRECT_DATATYPE:
                    dataset = incorrect_datatype(dataset, severity)
                case CorruptionTypes.INCONSISTENT_FORMAT:
                    dataset = inconsistent_format(dataset, severity)
                case CorruptionTypes.SHUFFLE_COLUMNS:
                    dataset = shuffle_columns(dataset, severity)
        corrupt_datasets.append(dataset)
    return corrupt_datasets


def analyze_dataset(
    gold_standard: pd.DataFrame,
    corrupted_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze a dataset for a given corruption type and severity.
    """
    pass


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
