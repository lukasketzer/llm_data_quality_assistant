"""call corrupt_dataset(
gold_standard: pd.DataFrame,
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
import numpy as np
from llm_data_quality_assistant.corruptor import (
    corrupt_dataset,
)
from llm_data_quality_assistant.enums.CorruptionTypes import (
    RowCorruptionTypes,
    CellCorruptionTypes,
)

from llm_data_quality_assistant.llm_integration import merge_datasets_by_primary_key
from llm_data_quality_assistant.enums import Models
from llm_data_quality_assistant.evaluation import (
    evaluate_dataset_micro,
    evaluate_dataset_macro,
)


class Pipeline:

    # TODO: primary key combined out of multiple parital keys

    @staticmethod
    def standardize_dataset(
        dataset: pd.DataFrame, primary_key: list[str]
    ) -> pd.DataFrame:
        if len(primary_key) == 0:
            raise ValueError("Primary key list cannot be empty.")

        missing_keys = [col for col in primary_key if col not in dataset.columns]
        if missing_keys:
            raise KeyError(
                f"Primary key columns {missing_keys} not found in dataset columns."
            )
        if dataset.empty:
            raise ValueError("Input dataset is empty.")

        return pd.concat(
            [group for _, group in dataset.groupby(primary_key)], ignore_index=True
        )

    @staticmethod
    # TODO: make it create parker-like-datasets
    def generate_corrupted_datasets(
        dataset: pd.DataFrame,
        cell_corruption_types: list[CellCorruptionTypes],
        row_corruption_types: list[RowCorruptionTypes],
        columns_to_exclude: list[str] = [],
        severity: float = 0.1,
        output_size: int = 5,
    ) -> list[pd.DataFrame]:
        corrupted_datasets, _ = corrupt_dataset(
            dataset=dataset,
            cell_corruption_types=cell_corruption_types,
            row_corruption_types=row_corruption_types,  # Add this argument as required
            columns_to_exclude=columns_to_exclude,
            severity=severity,
            output_size=output_size,
        )
        return corrupted_datasets

    @staticmethod
    def merge_with_llm(
        dataset: pd.DataFrame,
        primary_key: str,
        model_name: (
            Models.GeminiModels | Models.OllamaModels | Models.OpenAIModels
        ) = Models.GeminiModels.GEMINI_2_0_FLASH,
        rpm: int = 0,  # Requests per minute, 0 for no limit
        additional_prompt: str = "",
        verbose: bool = False,
        status_bar: bool = False,
    ) -> pd.DataFrame:
        return merge_datasets_by_primary_key(
            model_name=model_name,
            primary_key=primary_key,
            dataset=dataset,
            rpm=rpm,
            additional_prompt=additional_prompt,
            verbose=verbose,
            status_bar=status_bar,
        )

    @staticmethod
    def evaluate_micro(
        gold_standard: pd.DataFrame,
        cleaned_dataset: pd.DataFrame,
        corrupted_dataset: pd.DataFrame,
    ) -> dict:
        """Evaluate a generated dataset using micro metrics."""
        return evaluate_dataset_micro(
            gold_standard=gold_standard,
            cleaned_dataset=cleaned_dataset,
            corrupted_dataset=corrupted_dataset,
        )

    @staticmethod
    def evaluate_macro(
        gold_standard: pd.DataFrame,
        cleaned_dataset: pd.DataFrame,
        corrupted_dataset: pd.DataFrame,
    ) -> dict:
        """Evaluate a generated dataset using macro metrics."""
        return evaluate_dataset_macro(
            gold_standard=gold_standard,
            cleaned_dataset=cleaned_dataset,
            corrupted_dataset=corrupted_dataset,
        )
