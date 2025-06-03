import pandas as pd
from .llm_models import get_model
from .enums import Models
from .helper_functions.csv_helper import extract_csv_from_prompt


def merge_datasets_group_by_rows(
    model_name, datasets: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merges multiple datasets about the same thing using an LLM to resolve errors and output the most likely true values.
    """
    model = get_model(model_name)
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
    You are a CSV Merger tool designed to intelligently combine multiple rows of data from different versions of a dataset into a single, cohesive row.
    Your task is to analyze the provided rows and merge them by logically selecting the most appropriate values for each cell based on given entries.
    Your role is like that of a data analyst with strong reasoning skills.
    You need to ensure that for every attribute, you assess the values critically and decide which one to keep, especially when there are conflicts.
    You're adept at recognizing patterns and choosing defaults when necessary, aiming for accuracy and completeness in the output.
    The audience for your output includes software developers and data scientists who need to process and consolidate CSV files.
    They rely on you for accurate merging of datasets to facilitate easier data analysis and reporting.
    Your task is to take input rows formatted as CSV text and produce a single, merged CSV output.
    Each row you receive may have missing or varying values, which you will evaluate to return a valid merged CSV row with the best values selected logically.

    IMPORTANT: Output ONLY the merged dataset as a valid CSV string, with the same columns as the input.
    DO NOT include any explanations, markdown, code blocks, or extra formatting—output ONLY the CSV data.
    If you include anything other than the CSV, the production process will fail.

   EXAMPLE INPUT:

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

    EXAMPLE OUTPUT:

    1,John Doe,john.doe@example.com,30,New York  
    2,Jane Smith,jane.smith@example.com,28,Los Angeles


    Here are the datasets to merge grouped by rows:
    {cols}
    {csvs.strip()}
    """
    # prompt = (
    #     "You are a CSV data merging assistant. "
    #     "You will be given multiple datasets about the same topic, but they may contain errors or inconsistencies. "
    #     "Your job is to go through the datasets row by row and compare the rows of each version of the dataset. "
    #     "Use logial reasoning to determine the most likely true value for each cell, and create one clean row out of the multiple corrupted versions."
    #     "Repeat this for each row in the datasets. "
    #     f"For each row, you will be given {len(datasets)} versions of the dataset, each with {n_cols} columns: {', '.join(cols)}. "
    #     "Your task is to merge them into a single dataset, choosing the most likely true value for each cell. "
    #     "IMPORTANT: Output ONLY the merged dataset as a valid CSV string, with the same columns as the input. "
    #     "DO NOT include any explanations, markdown, code blocks, or extra formatting—output ONLY the CSV data. "
    #     "If you include anything other than the CSV, the production process will fail. "
    #     "Here are the datasets to merge grouped by rows:\n\n" + csvs.strip()
    # )
    # Get the merged dataset from the LLM
    message = ""
    merged_csv = model.generate(prompt, stream=True)
    for chunk in merged_csv:
        print(chunk, end="", flush=True)
        message += chunk

        # Try to parse the LLM output as a DataFrame
        # try:
    merged_df = extract_csv_from_prompt(message)
    # except Exception:
    #     merged_df = pd.DataFrame()  # fallback if parsing fails
    print("merged dataset:")
    print(merged_df)
    print(merged_df.shape)
    return merged_df


def merge_datasets_group_by_dataset(
    model_name, datasets: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merges multiple datasets about the same thing using an LLM to resolve errors and output the most likely true values.
    """
    model = get_model(model_name)

    # Combine all datasets into CSV strings for the prompt
    csvs = []
    for i, df in enumerate(datasets):
        csvs.append(f"Dataset {i+1}:\n" + df.to_csv(index=False))
    prompt = (
        "You are a CSV data merging assistant. "
        "You will be given multiple datasets about the same topic, but they may contain errors or inconsistencies. "
        "Your task is to merge them into a single dataset, choosing the most likely true value for each cell. "
        "IMPORTANT: Output ONLY the merged dataset as a valid CSV string, with the same columns as the input. "
        "DO NOT include any explanations, markdown, code blocks, or extra formatting—output ONLY the CSV data. "
        "If you include anything other than the CSV, the production process will fail. "
        "Here are the datasets to merge:\n\n" + "\n".join(csvs)
    )

    # Get the merged dataset from the LLM
    message = ""
    merged_csv = model.generate(prompt, stream=True)
    for chunk in merged_csv:
        print(chunk, end="", flush=True)
        message += chunk

    # Try to parse the LLM output as a DataFrame
    try:
        merged_df = extract_csv_from_prompt(message)
    except Exception:
        merged_df = pd.DataFrame()  # fallback if parsing fails
    return merged_df


def merge_dataset_in_chunks_with_llm(
    model_name, datasets: list[pd.DataFrame], chunk_size: int = 50
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
        merged_chunk = merge_datasets_group_by_rows(model_name, chunk_dfs)
        merged_chunks.append(merged_chunk)
    merged_df = pd.concat(merged_chunks, ignore_index=True)
    merged_df = merged_df.drop_duplicates(ignore_index=True)
    return merged_df

def merge_single_corrupted_dataset(
    model_name, dataset: pd.DataFrame
) -> pd.DataFrame:
    """
    Merges a single corrupted dataset using an LLM to resolve errors and output the most likely true values.
    """
    if dataset is None:
        return pd.DataFrame()
    
    model = get_model(model_name)

    # Combine all datasets into CSV strings for the prompt
    csv = dataset.to_csv(index=False)
    
    prompt = (
        "You are a CSV data merging assistant. "
        "You will a dataset about the same topic, but it may contain errors or inconsistencies. "
        "Your task is to clean it, choosing the most likely true value for each cell. "
        "IMPORTANT: Output ONLY the merged dataset as a valid CSV string, with the same columns as the input. "
        "DO NOT include any explanations, markdown, code blocks, or extra formatting—output ONLY the CSV data. "
        "If you include anything other than the CSV, the production process will fail. "
        "Every line has to have the same number of fields!!!"
        "Here is the dataset to clean:\n\n" + csv
    )

    # Get the merged dataset from the LLM
    message = ""
    merged_csv = model.generate(prompt, stream=True)
    for chunk in merged_csv:
        print(chunk, end="", flush=True)
        message += chunk

    # Try to parse the LLM output as a DataFrame
    merged_df = extract_csv_from_prompt(message)

    return merged_df

if __name__ == "__main__":
    pass
