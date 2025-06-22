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
    # TODO: multiple primary keys
    @staticmethod
    def standardize_datasets(
        primary_key: str,
        **datasets: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        if not datasets:
            return {}

        output = {name: df.copy() for name, df in datasets.items()}

        # Check if primary keys exist in all datasets
        if not all(primary_key in df.columns for df in output.values()):
            raise ValueError(
                f"Primary key '{primary_key}' must be present in all datasets."
            )

        # Check shapes
        shapes = [df.shape for df in output.values()]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All datasets must have the same shape.")

        # Check columns
        columns = [set(df.columns.tolist()) for df in output.values()]
        if not all(col == columns[0] for col in columns):
            raise ValueError("All datasets must have the same columns.")

        # Check available primary keys
        primary_keys = sorted(next(iter(output.values()))[primary_key].tolist())
        for name, dataset in output.items():
            if primary_keys != sorted(dataset[primary_key].tolist()):
                raise ValueError(
                    f"Primary key '{primary_key}' must have the same values in all datasets."
                )

        # Sort columns
        columns_order = next(iter(output.values())).columns
        output = {name: df[columns_order] for name, df in output.items()}

        # Sort primary keys
        output = {
            name: df.sort_values(by=primary_key).reset_index(drop=True)
            for name, df in output.items()
        }

        return output

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
