import pandas as pd
from llm_models import get_model
from enums import Models


def merge_dataset(model_name, datasets: list[pd.DataFrame]) -> pd.DataFrame:
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
        "Here are the datasets to merge:\n\n"
        + "\n".join(csvs)
    )

    # Get the merged dataset from the LLM
    merged_csv = model.generate(prompt)
    # Try to parse the LLM output as a DataFrame
    try:
        from io import StringIO
        merged_df = pd.read_csv(StringIO(merged_csv))
    except Exception:
        merged_df = pd.DataFrame()  # fallback if parsing fails
    return merged_df


if __name__ == "__main__":
    pass
