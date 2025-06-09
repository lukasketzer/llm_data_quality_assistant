import pandas as pd


def merge_baseline(primary_key: str, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline merging function that merges the dataset based on the primary key using a simple aggregation (mode).
    The function signature and usage are now consistent with the LLM-based merging functions.
    """
    if dataset is None:
        return pd.DataFrame()

    if primary_key not in dataset.columns:
        raise ValueError(f"Primary key '{primary_key}' not found in dataset columns.")

    # Group by the primary key and aggregate the values
    merged_df = (
        dataset.groupby(primary_key)
        .agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )

    # Ensure that the shape of the dataset does not change
    merged_df = merged_df.reindex(columns=dataset.columns, fill_value=None)

    return merged_df


def merge_multiple_datasets_baseline(
    primary_key: str, datasets: list[pd.DataFrame]
) -> pd.DataFrame:
    """
    Baseline merging function for a list of datasets. Concatenates the datasets and merges them by the primary key.
    The function signature and usage are consistent with the LLM-based merging functions.
    """
    if not datasets or len(datasets) == 0:
        return pd.DataFrame()

    merged_df = pd.concat(datasets, ignore_index=True)
    return merge_baseline(primary_key=primary_key, dataset=merged_df)
