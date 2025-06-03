"""call corrupt_dataset(
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


from pprint import pprint
import pandas as pd
from corruptor import corrupt_dataset, RowCorruptionTypes, CellCorruptionTypes
from llm_integration import merge_dataset_in_chunks_with_llm
from enums import Models
from evaluation import evaluate_dataset_micro, evaluate_dataset_macro


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
            row_corruption_types=row_corruption_type,
            cell_corruption_types=cell_corruption_type,
            severity=severity,
            output_size=output_size,
        )
        return corrupted_datasets, corrupted_coords

    def merge_with_llm(
        self,
        datasets: list[pd.DataFrame],
        model_name: (
            Models.GeminiModels | Models.OllamaModels | Models.OpenAIModels
        ) = Models.GeminiModels.GEMINI_2_0_FLASH,
    ):
        merged_df = merge_dataset_in_chunks_with_llm(
            model_name, datasets, chunk_size=50
        )
        return merged_df

    def evaluate_micro(self, generated_dataset, corrupted_coords):
        """Evaluate a generated dataset using micro metrics."""
        return evaluate_dataset_micro(self.dataset, generated_dataset, corrupted_coords)

    def evaluate_macro(self, generated_dataset, corrupted_coords):
        """Evaluate a generated dataset using macro metrics."""
        return evaluate_dataset_macro(self.dataset, generated_dataset, corrupted_coords)


# Example usage:
if __name__ == "__main__":
    gold_standard = pd.read_csv(
        "datasets/parker_datasets/gold_standard_alergene_pivoted.csv"
    ).head(
        50
    )  # Load a sample of the dataset
    pipeline = Pipeline(gold_standard)

    row_corruption_types = []
    cell_corruption_types = [CellCorruptionTypes.OUTLIER, CellCorruptionTypes.NULL]
    print("Corruptig datasets....")
    corrupted_datasets, corrupted_coords = pipeline.generate_corrupted_datasets(
        row_corruption_type=row_corruption_types,
        cell_corruption_type=cell_corruption_types,
        severity=0.15,
        output_size=5,
    )
    print("Corruptig done.")

    # Print each corrupted dataset with a blank line in between
    # print("Corrupted datasets:")
    # for i, df in enumerate(corrupted_datasets):
    #     print(f"Dataset {i+1}:")
    #     print(df)
    #     print()  # Blank line for separation

    # Example: merge the corrupted datasets using LLM
    print("Start merging...")
    merged_df = pipeline.merge_with_llm(corrupted_datasets)
    print("Merging done.")
    # print("\n")
    # print("Merged dataset:")
    # print(merged_df)
    # print()
    stats = pipeline.evaluate_micro(merged_df, corrupted_coords)
    print("Micro evaluation stats:")
    pprint(stats)
    # # Evaluate each corrupted dataset
    # for i, (df, coords) in enumerate(zip(corrupted_datasets, corrupted_coords)):
    #     print(f"Evaluation for Dataset {i+1}:")
    #     micro_stats = pipeline.evaluate_micro(df, coords)
    #     macro_stats = pipeline.evaluate_macro(df, coords)
    #     print("Micro stats:")
    #     pprint(micro_stats)
    #     print()
    #     #print("Macro stats:")
    #     #pprint(macro_stats)
