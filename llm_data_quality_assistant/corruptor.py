import numpy as np
from llm_data_quality_assistant.enums.CorruptionTypes import (
    RowCorruptionTypes,
    CellCorruptionTypes,
)
import pandas as pd
import math
from llm_data_quality_assistant.corruption_functions import *


datatype_restrictions = {
    CellCorruptionTypes.ROUNDING_ERROR: [
        np.dtype("int64"),
        np.dtype("float64"),
    ],
    CellCorruptionTypes.CASE_ERROR: [np.dtype("O")],
    CellCorruptionTypes.TRUNCATE: [np.dtype("O")],
    CellCorruptionTypes.TYPO: [np.dtype("O")],
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
    dataset: pd.DataFrame,
    row_corruption_type: list[RowCorruptionTypes],
    severity: float,
    excluded_coords: np.ndarray = np.empty((0, 2), dtype=int),
) -> dict[RowCorruptionTypes, np.ndarray]:
    if len(row_corruption_type) == 0:
        return {}

    dataset_dimensions = dataset.shape

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
    dataset: pd.DataFrame,
    cell_corruption_type: list[CellCorruptionTypes],
    severity: float,
    excluded_coords: np.ndarray = np.empty((0, 2), dtype=int),
) -> dict[CellCorruptionTypes, np.ndarray]:

    dataset_dimensions: tuple[int, int] = dataset.shape

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
    dataset: pd.DataFrame,
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

    dataset_dimensions: tuple[int, int] = dataset.shape

    percentage_rows = (
        len(row_corruption_type)
        / (len(row_corruption_type) + len(cell_corruption_type))
        if (len(row_corruption_type) + len(cell_corruption_type)) > 0
        else 0
    )
    percentage_cells = 1 - percentage_rows
    row_corruption_config = calculate_row_corruption(
        dataset=dataset,
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
        dataset=dataset,
        cell_corruption_type=cell_corruption_type,
        severity=severity * percentage_cells,
        excluded_coords=row_coords,
    )

    cell_coords = np.array(
        [
            [row, col]
            for coords in cell_corruption_config.values()
            for (row, col) in coords
        ],
        dtype=int,
    )

    if cell_coords.shape[0] > 0 and total_coordinates.shape[0] > 0:
        total_coordinates = np.vstack([total_coordinates, cell_coords])
    elif cell_coords.shape[0] > 0 and total_coordinates.shape[0] == 0:
        total_coordinates = cell_coords

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
    dataset: pd.DataFrame,
    row_corruption_types: list[RowCorruptionTypes],
    cell_corruption_types: list[CellCorruptionTypes],
    columns_to_exclude: list[str] = [],
    severity: float = 0.1,  # Severity of corruption (0.0 to 1.0)
    output_size: int = 5,
) -> tuple[list[pd.DataFrame], list[np.ndarray]]:
    """
    Apply a corruption type to the dataset with a given severity.
    """

    for col in columns_to_exclude:
        if col not in dataset.columns:
            raise ValueError(f"Column '{col}' not found in the dataset.")

    dtypes = set(dataset.dtypes.values)

    for c in cell_corruption_types + row_corruption_types:
        if c in datatype_restrictions:
            if not any(dtype in dtypes for dtype in datatype_restrictions[c]):
                raise ValueError(
                    f"Cell corruption type {c} is not applicable to the dataset's data types."
                )

    # Remove excluded columns from the dataset
    original_column_order = dataset.columns
    dataset_to_corrupt = dataset.copy()
    if len(columns_to_exclude) > 0:
        dataset_to_corrupt = dataset_to_corrupt.drop(
            columns_to_exclude, axis=1, errors="ignore"
        )

    corrupted_datasets = []
    corrupted_coords = []

    for _ in range(output_size):
        dataset_copy = dataset_to_corrupt.copy()
        row_corruption_config, cell_corruption_config, total_coordinates = (
            calculate_corruption(
                dataset=dataset_copy,
                row_corruption_type=row_corruption_types,
                cell_corruption_type=cell_corruption_types,
                severity=severity,
            )
        )
        dataset_dtypes = dataset_copy.dtypes
        dataset_copy = dataset_copy.astype(object)

        dataset_copy = apply_row_corruptions(
            row_corruption_config=row_corruption_config,
            dataset=dataset_copy,
        )
        dataset_copy = apply_cell_corruptions(
            cell_corruption_config=cell_corruption_config,
            dataset=dataset_copy,
        )

        # Reinsert the excluded columns
        for col in columns_to_exclude:
            dataset_copy[col] = dataset[col]

        # Ensure the original column order is maintained
        dataset_copy = dataset_copy[original_column_order]

        for col in columns_to_exclude:
            col_idx = dataset_copy.columns.get_loc(col)
            # Adjust coordinates to account for excluded columns
            total_coordinates[:, 1] = np.where(
                total_coordinates[:, 1] < col_idx,
                total_coordinates[:, 1],
                total_coordinates[:, 1] + 1,
            )

        corrupted_coords.append(total_coordinates)

        corrupted_datasets.append(dataset_copy)

    return corrupted_datasets, corrupted_coords
