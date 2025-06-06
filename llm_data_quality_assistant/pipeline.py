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
from .corruptor import corrupt_dataset, RowCorruptionTypes, CellCorruptionTypes
from .llm_integration import (
    merge_dataset_in_chunks_with_llm,
    merge_single_corrupted_dataset,
)
from .enums import Models
from .evaluation import evaluate_dataset_micro, evaluate_dataset_macro


# TODO: Architekutr überdenken
class Pipeline:
    def __init__(self, gold_standard: pd.DataFrame):
        self.dataset = gold_standard

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
        chunk_size: int = 50,
        group_by_rows: bool = False,
        additional_prompt: str = "",
        verbose: bool = False,
    ):
        merged_df = merge_dataset_in_chunks_with_llm(
            model_name,
            datasets,
            chunk_size=chunk_size,
            group_by_rows=group_by_rows,
            additional_prompt=additional_prompt,
            verbose=verbose,
        )
        return merged_df

    # TODO: hält sich nicht an die architektur
    # mit "merge_with_llm" zuammenführen eventuell
    def clean_single_dataset(
        self,
        dataset: pd.DataFrame,
        additional_prompt: str = "",
        verbose: bool = False,
        model_name: (
            Models.GeminiModels | Models.OllamaModels | Models.OpenAIModels
        ) = Models.GeminiModels.GEMINI_2_0_FLASH,
    ):
        merged_df = merge_single_corrupted_dataset(model_name, dataset, additional_prompt=additional_prompt, verbose=verbose)
        return merged_df

    def evaluate_micro(self, generated_dataset, corrupted_coords):
        """Evaluate a generated dataset using micro metrics."""
        return evaluate_dataset_micro(self.dataset, generated_dataset, corrupted_coords)

    def evaluate_macro(self, generated_dataset, corrupted_coords):
        """Evaluate a generated dataset using macro metrics."""
        return evaluate_dataset_macro(self.dataset, generated_dataset, corrupted_coords)
