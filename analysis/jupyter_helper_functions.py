# jupyter_helper_functions.py
"""
Reusable helper functions for Jupyter notebooks in the data quality project.
"""

import pandas as pd
import string
import json
import os
from pprint import pprint
import time
from typing import Optional, Any
from llm_data_quality_assistant.pipeline import Pipeline


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename (replace punctuation and spaces with underscores)."""
    for p in string.punctuation + " ":
        name = name.replace(p, "_")
    return name


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset from the given path."""
    return pd.read_csv(path)


def load_text_file(path: str) -> str:
    """Load a text file (e.g., rules or partial keys)."""
    with open(path, "r") as f:
        return f.read()


def get_unique_filename(path: str) -> str:
    """
    If the file exists, append a number before the extension until a unique filename is found.
    """
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    new_path = f"{base}_{i}{ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base}_{i}{ext}"
    return new_path


def save_json(data, path: str) -> None:
    """Save a Python object as a JSON file, avoiding overwrite by appending a number if needed."""
    path = get_unique_filename(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_dataframe_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as a CSV file, avoiding overwrite by appending a number if needed."""
    path = get_unique_filename(path)
    df.to_csv(path, index=False)


def merge_with_llm_timed(
    dataset: pd.DataFrame,
    primary_key: str,
    model: Any,
    rpm: int = 30,
    additional_prompt: Optional[str] = None,
) -> tuple[pd.DataFrame, float]:
    """
    Merge a dataset with LLM and return (merged DataFrame, elapsed time in seconds).
    """
    start_time = time.time()
    merged_df = Pipeline.merge_with_llm(
        dataset=dataset,
        primary_key=primary_key,
        model_name=model,
        rpm=rpm,
        additional_prompt=additional_prompt if additional_prompt is not None else "",
        verbose=False,
        status_bar=True,
    )
    elapsed = time.time() - start_time
    return merged_df, elapsed


def standardize_and_evaluate(
    gold_standard: pd.DataFrame,
    merged_df: pd.DataFrame,
    corrupt_dataset: pd.DataFrame,
    primary_key: str,
    time_delta: float,
    results_dir: str,
    file_name: str,
) -> None:
    """
    Standardize datasets, evaluate, and save results.
    """
    out = Pipeline.standardize_datasets(
        gold_standard=gold_standard,
        cleaned_dataset=merged_df,
        corrupted_dataset=corrupt_dataset,
        primary_key=primary_key,
    )
    gold_standard = out["gold_standard"]
    merged_df = out["cleaned_dataset"]
    corrupt_dataset = out["corrupted_dataset"]

    stats_micro = Pipeline.evaluate_micro(
        gold_standard=gold_standard,
        cleaned_dataset=merged_df,
        corrupted_dataset=corrupt_dataset,
    )
    stats_micro["time_taken"] = time_delta

    stats_macro = Pipeline.evaluate_macro(
        gold_standard=gold_standard,
        cleaned_dataset=merged_df,
        corrupted_dataset=corrupt_dataset,
    )
    stats_macro["time_taken"] = time_delta

    save_json(stats_micro, f"{results_dir}{file_name}_results_micro.json")
    save_json(stats_macro, f"{results_dir}{file_name}_results_macro.json")
    pprint(stats_micro)
    pprint(stats_macro)
