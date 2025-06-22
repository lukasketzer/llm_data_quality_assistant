import pandas as pd
from llm_data_quality_assistant.llm_models import get_model
from pydantic import create_model
import json
from llm_data_quality_assistant.merge_baseline import merge_baseline
from llm_data_quality_assistant.llm_models import OpenAIModel
import time
from tqdm import tqdm
from typing import List
from pydantic import RootModel
import pandas as pd
import numpy as np


dtype_map = {
    "int64": int,
    "float64": float,
    "object": str,
    "bool": bool,
}


def combine_results_1(
    df_parker: pd.DataFrame, df_llm: pd.DataFrame, df_original: pd.DataFrame
) -> pd.DataFrame:
    """
    Combines two DataFrames by merging them on their index using a majority vote per cell.
    If all three disagree, defaults to df_parker (assumed more accurate).
    """
    df_parker = df_parker.reset_index(drop=True)
    df_llm = df_llm.reset_index(drop=True)
    df_original = df_original.reset_index(drop=True)

    if df_parker.shape != df_llm.shape or df_parker.shape != df_original.shape:
        raise ValueError("All DataFrames must have the same shape after index reset.")

    df_cleaned = pd.DataFrame(index=df_parker.index, columns=df_parker.columns)

    for row in df_parker.index:
        for col in df_parker.columns:
            parker_val = df_parker.at[row, col]
            llm_val = df_llm.at[row, col]
            orig_val = df_original.at[row, col]
            # Majority vote logic
            if parker_val == llm_val or parker_val != orig_val:
                df_cleaned.at[row, col] = parker_val
            elif llm_val != orig_val:
                df_cleaned.at[row, col] = llm_val
            else:
                # All three disagree, default to parker_val
                df_cleaned.at[row, col] = orig_val

    """
    for row in df_parker.index:
        for col in df_parker.columns:
            parker_val = df_parker.at[row, col]
            llm_val = df_llm.at[row, col]
            orig_val = df_original.at[row, col]
            if parker_val == llm_val:
                df_cleaned.at[row, col] = parker_val
            elif parker_val != orig_val and llm_val != orig_val:
                df_cleaned.at[row, col] = llm_val
            elif parker_val != orig_val:
                df_cleaned.at[row, col] = parker_val
            elif llm_val != orig_val:
                df_cleaned.at[row, col] = llm_val
            else:
                df_cleaned.at[row, col] = orig_val 
    """

    return df_cleaned


def __generate_pydantic_structure(dataset: pd.DataFrame):
    datatypes = {}
    for col_name, d in dataset.dtypes.items():
        datatypes[col_name] = dtype_map[str(d)]
    return create_model("StructuredOutput", **datatypes)


def __merge_datasets(
    model_name, prompt, dataset: pd.DataFrame, verbose=False
) -> pd.DataFrame:
    struct = __generate_pydantic_structure(dataset=dataset)
    model = get_model(model_name)
    if not isinstance(model, OpenAIModel):

        class ListStruct(RootModel[list[struct]]):
            pass

        struct = ListStruct

    message = ""
    if verbose:
        output = model.generate_stream(prompt=prompt, format=struct)
        for chunk in output:
            print(chunk, end="", flush=True)
            message += chunk

    else:
        message = model.generate(
            prompt=prompt,
            format=struct,
        )
    data = json.loads(message)

    if isinstance(model, OpenAIModel):
        data = [data]
    return pd.DataFrame(data)


def merge_datasets_by_primary_key(
    model_name,
    primary_key: str,
    dataset: pd.DataFrame,
    verbose: bool = False,
    status_bar: bool = False,
    rpm: int = 0,  # Requests per minute, 0 for no limit
    additional_prompt: str = "",
) -> pd.DataFrame:
    """
    Merges rows in the dataset by the given primary key using the LLM-based cleaning approach.
    Returns a single DataFrame with all merged rows.
    """
    if dataset is None:
        return pd.DataFrame()

    if primary_key not in dataset.columns:
        raise ValueError(f"Primary key '{primary_key}' not found in dataset columns.")

    # Add a temporary index to restore the original order later
    dataset = dataset.copy()
    dataset["_tmp_row_order"] = range(len(dataset))

    min_interval = 60 / rpm if rpm > 0 else 0
    last_request_time = None

    grouped = [group for _, group in dataset.groupby(primary_key)]
    merged_rows = []

    for group in tqdm(grouped, desc="Merging groups with LLM", disable=not status_bar):
        now = time.time()
        if last_request_time is not None and min_interval > 0:
            elapsed = now - last_request_time
            to_wait = min_interval - elapsed
            if to_wait > 0:
                time.sleep(to_wait)

        last_request_time = time.time()

        row_order = group["_tmp_row_order"].values
        group.drop(columns=["_tmp_row_order"], inplace=True)

        merged_row = __merge_single_corrupted_dataset(
            model_name=model_name,
            dataset=group,
            additional_prompt=additional_prompt,
            verbose=verbose,
        )
        merged_row["_tmp_row_order"] = row_order
        merged_rows.append(merged_row)

    if merged_rows:
        # TODO: maybe a better way to merge DataFrames?
        merged_df = pd.concat(merged_rows, ignore_index=True)
        merged_df = merged_df.reindex(columns=dataset.columns)

        for col in dataset.columns:
            if pd.api.types.is_integer_dtype(dataset[col].dtype):
                # Use pandas nullable Int64 type to allow NA
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
                merged_df[col] = merged_df[col].astype("Int64")
            else:
                merged_df[col] = merged_df[col].astype(dataset[col].dtype)

        # Restore original order using the temporary index
        merged_df = (
            merged_df.sort_values("_tmp_row_order")
            .drop(columns=["_tmp_row_order"])
            .reset_index(drop=True)
        )

        assert (
            merged_df.shape[0] == dataset.shape[0]
        ), f"Expected {dataset.shape[0]} rows, but got {merged_df.shape[0]} rows after merging."

        assert list(dataset[primary_key].values) == list(
            merged_df[primary_key].values
        ), "Primary key values do not match or are not in the same order"

        return merged_df
    else:
        return pd.DataFrame()


# TODO: handle multiple rows at once
def __merge_single_corrupted_dataset(
    model_name,
    dataset: pd.DataFrame,
    additional_prompt: str = "",
    verbose=False,
) -> pd.DataFrame:
    if dataset is None:
        return pd.DataFrame()

    csv = dataset.to_csv(index=False)

    prompt = f"""
    You are a data cleaning assistant. 
    You will be given a dataset about the same topic, but it may contain errors or inconsistencies. 
    Your task is to clean it, choosing the most likely correct value for each cell.
    At first look at the different column names and find an identifier.
    Rows that have the same identifier should have the exact same values.
    After merging rows, you have to ensure that the values make sense. Think for yourself whether the values make sense or have to be changed. If you find a value that does not make sense, change it to a value that makes sense.
    Look for patterns in the additional information and use them to clean the dataset. Also use our own knowledge about the topic to clean the dataset.

    OUTPUT FORMAT: All of the inputted rows should be merged into a single row, with the same columns as the input dataset.

    IMPORTANT: Output ONLY the cleaned dataset as a valid JSON array of objects, with the same columns as the input. 
    DO NOT include any explanations, markdown, code blocks, or extra formatting—output ONLY the JSON data. 
    If you include anything other than the JSON, the production process will fail. 

    {"Here is some additional information to help you merge the datasets. Use these information to ensure that all the values in the dataset are valid and make sense. The values in the dataset have to adhere to the additional information" if additional_prompt != "" else ""}
    {additional_prompt.strip()}

    Here is the dataset to clean (as CSV):

    {csv}
    """

    merged_df = __merge_datasets(
        model_name=model_name,
        dataset=dataset,
        prompt=prompt,
        verbose=verbose,
    )
    if merged_df.shape[0] == 0:
        raise ValueError(
            "The merged DataFrame is empty. This might be due to the LLM not returning any data. Please check the model and prompt."
        )
    else:
        # Take the first row and copy it n_row times to preserve the original shape
        return pd.concat([merged_df.iloc[[0]]] * len(dataset), ignore_index=True)
