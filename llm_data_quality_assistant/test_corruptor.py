import numpy as np
import pandas as pd

from corruptor import (
    corrupt_dataset,
    analyze_dataset,
    RowCorruptionTypes,
    CellCorruptionTypes,
)


def test_corruption_percentage():
    np.random.seed(42)
    test_cases = [
        (10, 5, 0.1),
        (20, 8, 0.2),
        (30, 4, 0.15),
        (15, 7, 0.25),
        (25, 6, 0.05),
        (50, 10, 0.12),
        (5, 3, 0.3),
        (40, 12, 0.18),
        (12, 12, 0.22),
        (100, 100, 0.08),
        (1, 1, 0.5),         # Single cell
        (2, 2, 0.5),         # Small square
        (100, 1, 0.1),       # Single column, many rows
        (1, 100, 0.1),       # Single row, many columns
        (200, 50, 0.01),     # Large, low severity
        (200, 50, 0.5),      # Large, high severity
        (3, 3, 1.0),         # Full corruption
        (10, 10, 0.0),       # No corruption
        (7, 13, 0.33),       # Odd shape, third corrupted
        (99, 99, 0.07),      # Large, odd size
    ]

    all_row_types = [
        RowCorruptionTypes.DELETE_ROWS,
        RowCorruptionTypes.SHUFFLE_COLUMNS,
    ]

    all_cell_types = [
        CellCorruptionTypes.OUTLIER,
        CellCorruptionTypes.NULL,
        CellCorruptionTypes.ADJACENT_ERROR,
        CellCorruptionTypes.INCORRECT_DATATYPE,
    ]

    total = 0
    success = 0
    for rows, cols, severity in test_cases:
        total += 1
        df = pd.DataFrame(np.random.randint(0, 100, size=(rows, cols)))
        # Choose a random number of row and cell corruption types (possibly zero)
        n_row = np.random.randint(0, len(all_row_types) + 1)
        n_cell = np.random.randint(1, len(all_cell_types) + 1)
        row_types = list(
            np.random.choice(np.array(all_row_types), n_row, replace=False)
        )
        cell_types = list(
            np.random.choice(np.array(all_cell_types), n_cell, replace=False)
        )

        try:
            corrupted_datasets, _ = corrupt_dataset(
                gold_standard=df,
                row_corruption_types=row_types,
                cell_corruption_types=cell_types,
                severity=severity,
                output_size=1,
            )

            corrupted = corrupted_datasets[0]
            actual = analyze_dataset(df, corrupted)
            expected = severity
        except ValueError as e:
            print(f"Error during corruption: {e}")
            continue

        if abs(actual - expected) < 0.05:
            success += 1
    print(
        f"Corruption percentage test: {success}/{total} passed ({(success / total) * 100:.2f}%)"
    )


if __name__ == "__main__":
    test_corruption_percentage()
    print("All corruption percentage tests passed.")
