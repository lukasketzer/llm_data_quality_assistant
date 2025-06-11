import pandas as pd
from llm_data_quality_assistant.llm_models import get_model
from pydantic import create_model
import json
from llm_data_quality_assistant.merge_baseline import merge_baseline

dtype_map = {
    "int64": int,
    "float64": float,
    "object": str,
    "bool": bool,
}


def generate_pydantic_structure(dataset: pd.DataFrame):
    datatypes = {}
    for col_name, d in dataset.dtypes.items():
        datatypes[col_name] = dtype_map[str(d)]
    return create_model("StructuredOutput", **datatypes)


def merge_datasets(
    model_name, prompt, dataset: pd.DataFrame, verbose=False
) -> pd.DataFrame:
    struct = generate_pydantic_structure(dataset=dataset)
    model = get_model(model_name)

    message = ""
    if verbose:
        output = model.generate_stream(
            prompt=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[struct],
            },
        )
        for chunk in output:
            print(chunk, end="", flush=True)
            message += chunk

    else:
        message = model.generate(
            prompt=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[struct],
            },
        )
    data = json.loads(message)
    return pd.DataFrame(data)


def merge_datasets_by_primary_key(
    model_name,
    primary_key: str,
    dataset: pd.DataFrame,
    verbose=False,
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

    grouped = [group for _, group in dataset.groupby(primary_key)]
    merged_rows = []
    for group in grouped:
        merged_row = merge_single_corrupted_dataset(
            model_name=model_name,
            dataset=group,
            additional_prompt=additional_prompt,
            verbose=verbose,
        )
        merged_rows.append(merged_row)

    if merged_rows:
        merged_df = pd.concat(merged_rows, ignore_index=True)
        # Ensure output DataFrame has same columns and dtypes as input
        merged_df = merged_df.reindex(columns=dataset.columns)
        for col in dataset.columns:
            merged_df[col] = merged_df[col].astype(dataset[col].dtype)
        return merged_df
    else:
        return pd.DataFrame()


def merge_single_corrupted_dataset(
    model_name, dataset: pd.DataFrame, additional_prompt: str = "", verbose=False
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
    If you want to merge rows with the same identifier, don't delete one of them, just give those rows the same values. Do not delete any rows.
    After merging rows, you have to ensure that the values make sense. Think for yourself whether the values make sense or have to be changed. If you find a value that does not make sense, change it to a value that makes sense.
    THE SHAPE OF THE DATASET MUST NOT CHANGE, meaning the number of rows and columns must stay the same.
    IMPORTANT: Output ONLY the cleaned dataset as a valid JSON array of objects, with the same columns as the input. 
    DO NOT include any explanations, markdown, code blocks, or extra formatting—output ONLY the JSON data. 
    If you include anything other than the JSON, the production process will fail. 

    {"Here is some additional information to help you merge the datasets. Use these information to ensure that all the values in the dataset are valid and make sense. The values in the dataset have to adhere to the additional information" if additional_prompt != "" else ""}
    {additional_prompt.strip()}

    Here is the dataset to clean (as CSV):

    {csv}
    """

    merged_df = merge_datasets(
        model_name=model_name,
        dataset=dataset,
        prompt=prompt,
        verbose=verbose,
    )
    return merged_df
