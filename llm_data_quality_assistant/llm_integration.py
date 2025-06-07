import pandas as pd
from llm_data_quality_assistant.llm_models import get_model
from pydantic import create_model
import json

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
    model_name, prompt, datasets: list[pd.DataFrame], verbose=False
) -> pd.DataFrame:
    struct = generate_pydantic_structure(datasets[0])
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


def merge_datasets_group_by_rows(
    model_name, datasets: list[pd.DataFrame], additional_prompt: str = "", verbose=False
) -> pd.DataFrame:
    """
    Merges multiple datasets about the same thing using an LLM to resolve errors and output the most likely true values.
    """
    n_rows, n_cols = datasets[0].shape
    cols = datasets[0].columns.tolist()

    # Combine all datasets into CSV strings for the prompt
    csvs = ""
    for row in range(n_rows):
        line = f"""
            Row {row + 1}:\n"""
        for dataset in datasets:
            line += dataset.iloc[[row]].to_csv(index=False, header=False)
        csvs += line.strip() + "\n"

    prompt = f"""
        You are a Dataset Merger tool designed to intelligently consolidate multiple variations of the same data row into a single, accurate, and complete version.
        You act like a data analyst with strong reasoning and pattern-recognition skills.

        ## OBJECTIVE:
        Given a set of grouped rows (variations of the same record),
        your task is to produce **one merged row per group**,
        where each cell contains the most appropriate and logically selected value across all inputs.
        Not following the rules will result in a lot of errors that might cause the production process to fail and lose a lot of money.

        ## FORMAT:
        - The input is a header followed by several rows, grouped per record.
        - The output must be **valid JSON**, formatted as a list of dictionaries—**one dictionary per row group**.
        - Each dictionary represents a single, fully-merged row with column names as keys.

        ## RULES FOR MERGING:
        - For each column, choose the most complete, accurate, and contextually appropriate value.
        - Prefer full names/emails over partial ones.
        - Choose non-null values where possible.
        - Resolve conflicts logically (e.g., “New York” > “NY” > "").
        - Do not drop or duplicate any row groups—**output must have the same number of rows as input groups**.

        ## AUDIENCE:
        Your output is consumed by software developers and data scientists who will use the resulting JSON for further automation, analysis, or reporting.
        Accuracy and consistency are critical.

        {"Here is some additional information to help you merge the datasets:" if additional_prompt != "" else ""}
        {additional_prompt.strip()}

        ## INPUT EXAMPLE:
        Row 1:  
        id,name,email,age,city  
        1,John Doe,john@example.com,30,New York  
        1,John,john.doe@example.com,,New York  
        1,John Doe,,30,  
        1,,john.d@example.com,30,New York  

        Row 2:  
        id,name,email,age,city  
        2,Jane Smith,jane.smith@example.com,28,Los Angeles  
        2,Jane Smith,,28,LA  
        2,Jane S.,jane@example.com,,Los Angeles  
        2,Jane,,28,Los Angeles  

        ## OUTPUT EXAMPLE (JSON):
        [
        {{
            "id": "1",
            "name": "John Doe",
            "email": "john@example.com",
            "age": "30",
            "city": "New York"
        }},
        {{
            "id": "2",
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "age": "28",
            "city": "Los Angeles"
        }}
        ]

        ## INPUT DATA:
        {cols}
        {csvs.strip()}
        """

    merged_df = merge_datasets(
        model_name=model_name,
        prompt=prompt,
        datasets=datasets,
        verbose=verbose,
    )

    return merged_df


def merge_datasets_group_by_dataset(
    model_name, datasets: list[pd.DataFrame], additional_prompt: str = "", verbose=False
) -> pd.DataFrame:
    """
    Merges multiple datasets about the same thing using an LLM to resolve errors and output the most likely true values.
    """

    # Combine all datasets into CSV strings for the prompt
    csvs = []
    for i, df in enumerate(datasets):
        csvs.append(f"Dataset {i+1}:\n" + df.to_csv(index=False))
    prompt = f"""
    You are a CSV data merging assistant. 
    You will be given multiple datasets about the same topic, but they may contain errors or inconsistencies. 
    Your task is to merge them into a single dataset, choosing the most likely true value for each cell. 
    IMPORTANT: Output ONLY the merged dataset as a valid JSON array of objects, with the same columns as the input. 
    DO NOT include any explanations, markdown, code blocks, or extra formatting—output ONLY the JSON data. 
    If you include anything other than the JSON, the production process will fail. 

    {"Here is some additional information to help you merge the datasets:" if additional_prompt != "" else ""}
    {additional_prompt.strip()}

    Here are the datasets to merge (as CSV):

    {additional_prompt}
    {chr(10).join(csvs)}
    """

    # Get the merged dataset from the LLM
    merged_df = merge_datasets(
        model_name=model_name,
        prompt=prompt,
        datasets=datasets,
        verbose=verbose,
    )
    return merged_df


def merge_dataset_in_chunks_with_llm(
    model_name,
    datasets: list[pd.DataFrame],
    chunk_size: int = 50,
    additional_prompt: str = "",
    group_by_rows: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Splits the datasets into row-wise chunks, merges each chunk using merge_whole_dataset,
    and concatenates the results into a single DataFrame. Removes duplicate rows at the end.
    """
    if not datasets:
        return pd.DataFrame()
    num_rows = min(len(df) for df in datasets)
    merged_chunks = []
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk_dfs = [df.iloc[start:end] for df in datasets]
        merged_chunk = (
            merge_datasets_group_by_rows(
                model_name,
                chunk_dfs,
                additional_prompt=additional_prompt,
                verbose=verbose,
            )
            if group_by_rows
            else merge_datasets_group_by_dataset(
                model_name,
                chunk_dfs,
                additional_prompt=additional_prompt,
                verbose=verbose,
            )
        )
        merged_chunks.append(merged_chunk)
    merged_df = pd.concat(merged_chunks, ignore_index=True)
    merged_df = merged_df.drop_duplicates(ignore_index=True)
    return merged_df


def merge_single_corrupted_dataset(
    model_name, dataset: pd.DataFrame, additional_prompt: str = "", verbose=False
) -> pd.DataFrame:
    if dataset is None:
        return pd.DataFrame()

    csv = dataset.to_csv(index=False)

    prompt = f"""
    You are a data cleaning assistant. 
    You will be given a dataset about the same topic, but it may contain errors or inconsistencies. 
    Your task is to clean it, choosing the most likely true value for each cell.
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
        prompt=prompt,
        datasets=[dataset],
        verbose=verbose,
    )
    return merged_df
