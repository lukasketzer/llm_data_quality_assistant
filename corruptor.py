import numpy as np
from enum import Enum
import pandas as pd
import math
from corruption_functions import *


class RowCorruptionTypes(Enum):
    SWAP_ROWS = "swap_rows"
    DELETE_ROWS = "delete_rows"
    SHUFFLE_COLUMNS = "shuffle_columns"
    REVERSE_ROWS = "reverse_rows"


class CellCorruptionTypes(Enum):
    OUTLIER = "outlier"
    NULL = "null"
    INCORRECT_DATATYPE = "incorrect_datatype"
    INCONSISTENT_FORMAT = "inconsistent_format"  # TODO: Implement
    SWAP_CELLS = "swap_cells"
    CASE_ERROR = "case_error"
    TRUNCATE = "truncate"
    ROUNDING_ERROR = "rounding_error"
    ENCODING_ERROR = "encoding_error"  # TODO: Implement
    TYPO = "typo"


datatype_restrictions = {
    CellCorruptionTypes.ROUNDING_ERROR: [
        np.float64,
        np.int64,
    ],
    CellCorruptionTypes.CASE_ERROR: [np.object_],
    CellCorruptionTypes.TRUNCATE: [np.object_],
    CellCorruptionTypes.TYPO: [np.object_],
}


def delete_coords(
    coordinates: np.ndarray, coordinates_to_remove: np.ndarray
) -> np.ndarray:
    """
    Remove rows from possible_cell_records that match any row in coords_to_remove.
    """
    if coordinates_to_remove.shape[0] == 0:
        return coordinates
    idx_to_delete = []
    for coord in coordinates_to_remove:
        idx = np.where((coordinates == coord).all(axis=1))[0]
        idx_to_delete.extend(idx)
    return np.delete(coordinates, idx_to_delete, axis=0)


def calculate_row_corruption(
    dataset_dimensions: tuple[int, int],
    row_corruption_type: list[RowCorruptionTypes],
    severity: float,
    excluded_coords: np.ndarray = np.empty((0, 2), dtype=int),
) -> dict[RowCorruptionTypes, np.ndarray]:
    if len(row_corruption_type) == 0:
        return {}

    amount = max(math.ceil(dataset_dimensions[0] * severity), 1)

    # Exclude rows that are in excluded_coords
    excluded_rows: np.ndarray = np.unique([row for (row, col) in excluded_coords])
    possible_rows: np.ndarray = np.array(
        [row for row in range(dataset_dimensions[0]) if row not in excluded_rows]
    )

    # Ensure we don't sample more rows than available
    amount = min(amount, len(possible_rows))
    if amount == 0:
        return {c: np.array([], dtype=int) for c in row_corruption_type}

    row_indices = np.random.choice(possible_rows, amount, replace=False)

    splits: list[np.ndarray] = np.array_split(row_indices, len(row_corruption_type))
    row_indices_map: dict[RowCorruptionTypes, np.ndarray] = {}
    for c, split in zip(row_corruption_type, splits):
        if c not in row_indices_map:
            row_indices_map[c] = split
        else:
            row_indices_map[c] = np.vstack([row_indices_map[c], split])

    return row_indices_map


def calculate_cell_corruption(
    dataset_dimensions: tuple[int, int],
    cell_corruption_type: list[CellCorruptionTypes],
    severity: float,
    excluded_coords: np.ndarray = np.empty((0, 2), dtype=int),
) -> dict[CellCorruptionTypes, np.ndarray]:

    if len(cell_corruption_type) == 0:
        return {}

    amount = max(math.ceil(dataset_dimensions[0] * dataset_dimensions[1] * severity), 1)

    possible_cell_records = np.array(
        [
            [row, col]
            for row in range(dataset_dimensions[0])
            for col in range(dataset_dimensions[1])
        ],
        dtype=int,
    )
    possible_cell_records = delete_coords(possible_cell_records, excluded_coords)
    cell_coordinates_map: dict[CellCorruptionTypes, np.ndarray] = {}

    # TODO: handle case where ther isnt a divider
    cells_per_type: int = (
        amount // len(cell_corruption_type) if len(cell_corruption_type) > 0 else 0
    )
    extra: int = amount % len(cell_corruption_type)

    for c in cell_corruption_type:
        cells_per_type_this_run = cells_per_type
        if extra > 0:
            cells_per_type_this_run += 1
            extra -= 1

        if c in datatype_restrictions:
            # Filter possible cell records based on datatype restrictions
            viable_columns = [
                idx
                for idx, dtype in enumerate(dataset.dtypes)
                if dtype in datatype_restrictions[c]
            ]
            possible_cells_for_corruption = np.array(
                [
                    [row, col]
                    for (row, col) in possible_cell_records
                    if col in viable_columns
                ]
            )
        else:
            possible_cells_for_corruption = possible_cell_records

        # choose the random indices
        if possible_cells_for_corruption.shape[0] == 0 or cells_per_type_this_run == 0:
            cell_coordinates = np.empty((0, 2), dtype=int)
        else:
            chosen_indices = np.random.choice(
                possible_cells_for_corruption.shape[0],
                cells_per_type_this_run,
                replace=False,
            )
            cell_coordinates = possible_cells_for_corruption[chosen_indices]

        if c not in cell_coordinates_map:
            cell_coordinates_map[c] = cell_coordinates
        else:
            cell_coordinates_map[c] = np.vstack(
                [cell_coordinates_map[c], cell_coordinates]
            )

        possible_cell_records = delete_coords(possible_cell_records, cell_coordinates)

    return cell_coordinates_map


def calculate_corruption(
    dataset_dimensions: tuple[int, int],
    row_corruption_type: list[RowCorruptionTypes],
    cell_corruption_type: list[CellCorruptionTypes],
    severity: float,
) -> tuple[
    dict[RowCorruptionTypes, np.ndarray],
    dict[CellCorruptionTypes, np.ndarray],
    np.ndarray,
]:
    """
    Calculate the noise to be added to the dataset based on the severity.
    """

    percentage_rows = (
        len(row_corruption_type)
        / (len(row_corruption_type) + len(cell_corruption_type))
        if (len(row_corruption_type) + len(cell_corruption_type)) > 0
        else 0
    )
    percentage_cells = 1 - percentage_rows
    row_corruption_config = calculate_row_corruption(
        dataset_dimensions=dataset_dimensions,
        row_corruption_type=row_corruption_type,
        severity=severity * percentage_rows,
    )
    # Collect all row indices and convert them to 2D coordinates (row, all columns)
    row_coords = np.array(
        [
            [row, col]
            for indices in row_corruption_config.values()
            for row in indices
            for col in range(dataset_dimensions[1])
        ],
        dtype=int,
    )

    total_coordinates = row_coords

    cell_corruption_config = calculate_cell_corruption(
        dataset_dimensions=dataset_dimensions,
        cell_corruption_type=cell_corruption_type,
        severity=severity * percentage_cells,
        excluded_coords=row_coords,
    )

    cell_coords = np.array(
        [
            (row, col)
            for coords in cell_corruption_config.values()
            for (row, col) in coords
        ],
        dtype=int,
    )

    total_coordinates = (
        np.vstack([total_coordinates, cell_coords])
        if len(total_coordinates) != 0
        else cell_coords
    )

    assert np.unique(total_coordinates, axis=0).shape[0] == total_coordinates.shape[0]

    return (
        row_corruption_config,
        cell_corruption_config,
        total_coordinates,  # np.ndarray of all coordinates
    )


def apply_row_corruptions(
    row_corruption_config: dict[RowCorruptionTypes, np.ndarray],
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the specified row corruption types to the dataset using the provided row indices.
    """
    if len(row_corruption_config) == 0:
        return dataset

    for c, split in row_corruption_config.items():
        match c:
            case RowCorruptionTypes.SWAP_ROWS:
                dataset = swap_rows(dataset, rows_to_swap=split)
            case RowCorruptionTypes.DELETE_ROWS:
                dataset = delete_rows(dataset, rows_to_delete=split)
            case RowCorruptionTypes.SHUFFLE_COLUMNS:
                dataset = shuffle_columns(dataset, rows_to_shuffle=split)
            case RowCorruptionTypes.REVERSE_ROWS:
                dataset = reverse_rows(dataset, rows_to_reverse=split)
            case _:
                raise ValueError(f"Unknown corruption type: {c}")
    return dataset


def apply_cell_corruptions(
    cell_corruption_config: dict[CellCorruptionTypes, np.ndarray],
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the specified cell corruption types to the dataset using the provided cell coordinates.
    """
    if len(cell_corruption_config) == 0:
        return dataset

    for c, coords in cell_corruption_config.items():
        match c:
            case CellCorruptionTypes.OUTLIER:
                dataset = outlier(dataset, coords)
            case CellCorruptionTypes.NULL:
                dataset = null(dataset, coords)
            case CellCorruptionTypes.INCORRECT_DATATYPE:
                dataset = incorrect_datatype(dataset, coords)
            case CellCorruptionTypes.INCONSISTENT_FORMAT:
                dataset = inconsistent_format(dataset, coords)
            case CellCorruptionTypes.SWAP_CELLS:
                dataset = swap_cells(dataset, coords)
            case CellCorruptionTypes.CASE_ERROR:
                dataset = case_error(dataset, coords)
            case CellCorruptionTypes.TRUNCATE:
                dataset = truncate(dataset, coords)
            case CellCorruptionTypes.ROUNDING_ERROR:
                dataset = rounding_error(dataset, coords)
            case CellCorruptionTypes.ENCODING_ERROR:
                dataset = encoding_error(dataset, coords)
            case CellCorruptionTypes.TYPO:
                dataset = typo(dataset, coords)
            case _:
                raise ValueError(f"Unknown cell corruption type: {c}")
    return dataset


def corrupt_dataset(
    gold_standard: pd.DataFrame,
    row_corruption_types: list[RowCorruptionTypes],
    cell_corruption_types: list[CellCorruptionTypes],
    severity: float = 0.1,  # Severity of corruption (0.0 to 1.0)
    output_size: int = 5,
) -> tuple[list[pd.DataFrame], list[np.ndarray]]:
    """
    Apply a corruption type to the dataset with a given severity.
    """
    dtypes = set(gold_standard.dtypes.values)

    for c in cell_corruption_types + row_corruption_types:
        if c in datatype_restrictions:
            if not any(dtype in dtypes for dtype in datatype_restrictions[c]):
                raise ValueError(
                    f"Cell corruption type {c} is not applicable to the dataset's data types."
                )

    corrupted_datasets = []
    corrupted_coords = []
    for _ in range(output_size):
        dataset = gold_standard.copy()
        row_corruption_config, cell_corruption_config, total_coordinates = (
            calculate_corruption(
                dataset_dimensions=dataset.shape,
                row_corruption_type=row_corruption_types,
                cell_corruption_type=cell_corruption_types,
                severity=severity,
            )
        )
        dataset = dataset.astype(object)

        corrupted_coords.append(total_coordinates)

        dataset = apply_row_corruptions(
            row_corruption_config=row_corruption_config,
            dataset=dataset,
        )
        dataset = apply_cell_corruptions(
            cell_corruption_config=cell_corruption_config,
            dataset=dataset,
        )

        corrupted_datasets.append(dataset)
    return corrupted_datasets, corrupted_coords


def analyze_dataset(
    gold_standard: pd.DataFrame,
    corrupted_dataset: pd.DataFrame,
) -> float:
    """
    Analyze a dataset for a given corruption type and severity.
    """

    # print(gold_standard)
    # print(corrupted_dataset)

    n_total: int = gold_standard.shape[0] * gold_standard.shape[1]
    n_corrupted: int = (
        (gold_standard != corrupted_dataset).sum().sum()
    )  # Count the number of corrupted cells
    return n_corrupted / n_total


if __name__ == "__main__":
    # dataset = pd.read_csv("datasets/selfwritte_dataset/dataset.csv")
    dataset = pd.read_csv("datasets/public_dataset/wine.data")

    # Test 1: Only cell corruption (NULL)
    corrupted_datasets, corrupted_coords = corrupt_dataset(
        gold_standard=dataset,
        row_corruption_types=[RowCorruptionTypes.DELETE_ROWS],
        cell_corruption_types=[CellCorruptionTypes.NULL, CellCorruptionTypes.OUTLIER],
        severity=0.15,
        output_size=1,
    )
    for i, corrupted_dataset in enumerate(corrupted_datasets):
        print(f"Test 1 - Corrupted Dataset {i + 1}")
        print(
            f"Analysis: {analyze_dataset(dataset, corrupted_dataset) * 100:.2f}% of the dataset is corrupted."
        )
