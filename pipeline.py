""" call corrupt_dataset(
    gold_standard: pd.DataFrame,
    row_corruption_type: list[RowCorruptionTypes],
    cell_corruption_type: list[CellCorruptionTypes],
    severity: float = 0.1,  # Severity of corruption (0.0 to 1.0)
    output_size: int = 5,
    ) in corruptor.py to generate a corrupted dataset.
"""

"""
call merge_dataset() in llm_integration.py to merge multiple datasets using an LLM.
"""


import pandas as pd
from corruptor import corrupt_dataset, RowCorruptionTypes, CellCorruptionTypes
from llm_integration import merge_dataset
from enums import Models

class Pipeline:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def generate_corrupted_datasets(
        self,
        row_corruption_type: list[RowCorruptionTypes],
        cell_corruption_type: list[CellCorruptionTypes],
        severity: float = 0.1,
        output_size: int = 5,
    ):
        corrupted_datasets, corrupted_coords = corrupt_dataset(
            gold_standard=self.dataset,
            row_corruption_type=row_corruption_type,
            cell_corruption_type=cell_corruption_type,
            severity=severity,
            output_size=output_size,
        )
        return corrupted_datasets, corrupted_coords

    def merge_with_llm(self, datasets: list[pd.DataFrame], model_name = Models.GeminiModels.GEMINI_2_0_FLASH):
        print(datasets)
        merged_df = merge_dataset(model_name, datasets)
        print("/n")
        print("Merged dataset:")
        print("/n")
        print(merged_df)
        return merged_df

# Example usage:
if __name__ == "__main__":
    gold_standard = pd.read_csv("datasets/selfwritte_dataset/dataset.csv")
    pipeline = Pipeline(gold_standard)

    row_corruption_types = [RowCorruptionTypes.DELETE_ROWS]
    cell_corruption_types = [CellCorruptionTypes.INCORRECT_DATATYPE, CellCorruptionTypes.INCONSISTENT_FORMAT, CellCorruptionTypes.ADJACENT_ERROR]

    corrupted_datasets, corrupted_coords = pipeline.generate_corrupted_datasets(
        row_corruption_type=row_corruption_types,
        cell_corruption_type=cell_corruption_types,
        severity=0.1,
        output_size=5,
    )

    print("First corrupted dataset:")
    print(corrupted_datasets[0])

    # Example: merge the corrupted datasets using LLM
    merged_df = pipeline.merge_with_llm(corrupted_datasets)
    print("Merged dataset:")
    print(merged_df)