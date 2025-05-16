import Faker
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
def swap_rows(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


# Franzi
def delete_rows(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


# Franzi
def missing_values(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


# Franzi
def duplicate_rows(dataset: pd.DataFrame, severity: float) -> pd.DataFrame:
    pass


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
    rows = np.random.randint(0, dataset_dimensions[0], amount_rows_to_corrupt)
    row_coordinates = np.array(
        [(row, col) for row in rows for col in range(dataset_dimensions[1])]
    )

    # Cells
    x_coords = np.random.randint(0, dataset_dimensions[0], amount_cells_to_corrupt)
    y_coords = np.random.randint(0, dataset_dimensions[1], amount_cells_to_corrupt)

    # Combine into (x, y) pairs
    cell_coordinates = np.column_stack((x_coords, y_coords))

    # Merge row_coordinates and cell_coordinates
    merged_coordinates = np.vstack((row_coordinates, cell_coordinates))

    return (
        rows,  # np.ndarray of row indices
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
    pass
