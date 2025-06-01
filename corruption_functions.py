import random
import numpy as np
import pandas as pd
import math


# Tested
def swap_rows(dataset: pd.DataFrame, rows_to_swap: np.ndarray) -> pd.DataFrame:

    if rows_to_swap.size == 0:
        return dataset

    if rows_to_swap.shape[1] != 1:
        raise ValueError("rows_to_delete must be a 1D array of row indices.")

    n_rows = dataset.shape[0]
    if np.any(rows_to_swap >= n_rows):
        raise IndexError("Row indices out of bounds.")

    if rows_to_swap.size < 2:
        return dataset  # No swap needed if less than two rows are provided

    # Generate a derangement (no index stays in place)
    while True:
        perm = np.random.permutation(rows_to_swap)
        if not np.any(perm == rows_to_swap):
            dataset.iloc[rows_to_swap] = dataset.iloc[perm].values
            break

    return dataset


# Tested
def delete_rows(dataset: pd.DataFrame, rows_to_delete: np.ndarray) -> pd.DataFrame:

    if rows_to_delete.size == 0:
        return dataset

    if rows_to_delete.shape[1] != 1:
        raise ValueError("rows_to_delete must be a 1D array of row indices.")

    # Pick one row index from the provided list
    dataset.iloc[rows_to_delete] = None

    return dataset


# Tested
def shuffle_columns(dataset: pd.DataFrame, rows_to_shuffle: np.ndarray) -> pd.DataFrame:
    """
    Shuffle the columns for the specified rows in the dataset.
    """
    if rows_to_shuffle.size == 0:
        return dataset

    if rows_to_shuffle.shape[1] != 1:
        raise ValueError("rows_to_shuffle must be a 1D array of row indices.")

    for row in rows_to_shuffle:

        shuffeled_values = np.random.permutation(dataset.values[row][0])
        dataset.iloc[row] = shuffeled_values
    return dataset


# Tested
def outlier(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    if cell_coordinates.size == 0:
        return dataset
    if cell_coordinates.shape[1] != 2:
        raise ValueError("cell_coordinates must be a 2D array with shape (n, 2).")

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


# Tested
def null(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    if cell_coordinates.size == 0:
        return dataset
    if cell_coordinates.shape[1] != 2:
        raise ValueError("cell_coordinates must be a 2D array with shape (n, 2).")

    for row, col in cell_coordinates:
        dataset.iat[row, col] = None
    return dataset


# Tested
def typo(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    """
    Introduce a random typo (swap, delete, or replace a character) in string values at the specified cell coordinates.
    """
    if cell_coordinates.size == 0:
        return dataset
    if cell_coordinates.shape[1] != 2:
        raise ValueError("cell_coordinates must be a 2D array with shape (n, 2).")

    for row, col in cell_coordinates:
        if not isinstance(dataset.iat[row, col], str) or not dataset.iat[row, col]:
            continue

        value = dataset.iat[row, col]
        error_type = random.choice(["swap", "delete", "replace"])
        idx = random.randint(0, len(value) - 1)

        if error_type == "swap" and len(value) > 1 and idx < len(value) - 1:
            s_list = list(value)
            s_list[idx], s_list[idx + 1] = s_list[idx + 1], s_list[idx]
            dataset.iat[row, col] = "".join(s_list)
        elif error_type == "delete" and len(value) > 1:
            dataset.iat[row, col] = value[:idx] + value[idx + 1 :]
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
                replacement_idx = keyboard_flat.index(char) + random.choice([-1, 1])
                replacement_idx = max(0, min(replacement_idx, len(keyboard_flat) - 1))
                replacement = keyboard_flat[replacement_idx]
            else:
                replacement = random.choice("abcdefghijklmnopqrstuvwxyz")
            dataset.iat[row, col] = value[:idx] + replacement + value[idx + 1 :]
        else:
            # If none of the above, leave value unchanged
            pass
    return dataset


def incorrect_datatype(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray
) -> pd.DataFrame:
    """
    Change the datatype of the specified cells to an incorrect type.
    For example, replace a number with a string, or a string with a number.
    """
    # Handle numpy integer and floating types if available
    numpy_integer = getattr(np, "integer", ())
    numpy_floating = getattr(np, "floating", ())
    for row, col in cell_coordinates:
        value = dataset.iat[row, col]
        # If value is numeric, replace with a string
        if (
            isinstance(value, (int, float))
            or isinstance(value, numpy_integer)
            or isinstance(value, numpy_floating)
        ):
            dataset.iat[row, col] = "not_a_number"
        # If value is a string, replace with a number
        elif isinstance(value, str):
            dataset.iat[row, col] = random.randint(10000, 99999)
        # If value is a bool, replace with a string
        elif isinstance(value, bool):
            dataset.iat[row, col] = "True" if value else "False"
        # If value is a datetime, replace with a string
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            dataset.iat[row, col] = "not_a_date"
        # Otherwise, replace with a string
        else:
            dataset.iat[row, col] = "incorrect_type"
    return dataset


# TODO:
def inconsistent_format(
    dataset: pd.DataFrame, cell_coordinates: np.ndarray
) -> pd.DataFrame:
    return dataset


# Tested
def reverse_rows(dataset: pd.DataFrame, rows_to_reverse: np.ndarray) -> pd.DataFrame:
    """
    Reverse the order of the specified rows in the dataset.
    """
    if rows_to_reverse.size == 0:
        return dataset
    for row in rows_to_reverse:
        reversed_rows = dataset.iloc[row].values[0][::-1]
        dataset.iloc[row] = reversed_rows
    return dataset


# Tested
def swap_cells(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    """
    Swap the values of the specified cell coordinates in a cyclic manner.
    """
    if cell_coordinates.size < 2:
        return dataset
    if cell_coordinates.shape[1] != 2:
        raise ValueError("cell_coordinates must be a 2D array with shape (n, 2).")

    values = [dataset.iat[row, col] for row, col in cell_coordinates]
    while True:
        perm = np.random.permutation(values)

        if not np.any(perm == values):
            for (row, col), value in zip(cell_coordinates, perm):
                dataset.iat[row, col] = value
            break

    return dataset


# Tested
def case_error(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    """
    Randomly change the case of string values in the specified cells.
    """
    for row, col in cell_coordinates:
        value = dataset.iat[row, col]
        if isinstance(value, str):
            if value == "":
                continue
            while True:
                case_func = random.choice(
                    [str.upper, str.lower, str.title, str.swapcase]
                )
                new_value = case_func(value)
                if new_value != value:
                    dataset.iat[row, col] = new_value
                    break

        else:
            raise TypeError(
                f"Expected string value at ({row}, {col}), but got {type(value).__name__}."
            )
    return dataset


# Tested
def truncate(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    """
    Truncate string or numeric values in the specified cells.
    """
    for row, col in cell_coordinates:
        value = dataset.iat[row, col]
        if isinstance(value, str) and len(value) > 1:
            cut = random.randint(1, len(value) - 1)
            dataset.iat[row, col] = value[:cut]
        else:
            raise TypeError(
                f"Expected string value at ({row}, {col}), but got {type(value).__name__}."
            )
    return dataset


def rounding_error(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    """
    Introduce rounding errors to numeric values in the specified cells.
    """
    for row, col in cell_coordinates:
        value = dataset.iat[row, col]
        if isinstance(value, np.floating) or isinstance(value, np.integer):
            lower = (value // 10) * 10
            dataset.iat[row, col] = random.choice([lower, lower + 10, lower + 20])
    return dataset


# TODO:
def encoding_error(dataset: pd.DataFrame, cell_coordinates: np.ndarray) -> pd.DataFrame:
    return dataset


if __name__ == "__main__":
    # Example usage
    name: float = 1234.0
    dataset = pd.DataFrame(
        {"A": ["apple", "banana", "cherry"], "B": [1, 2, 3], "C": [4.5, 5.5, 6.5]}
    )
    dataset = dataset.astype(object)
    print(dataset)
    rounding_error(
        dataset, np.array([[0, 1], [1, 1], [2, 1]])
    )  # Should change the case of the string
    print(dataset)
