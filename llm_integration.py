import pandas as pd
from llm_models import get_model
from enums import Models


def merge_dataset(model_name, datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges the dataset with the model name.
    """
    model = get_model(model_name)

    # Placeholder for merging logic
    return pd.DataFrame()


if __name__ == "__main__":
    pass
