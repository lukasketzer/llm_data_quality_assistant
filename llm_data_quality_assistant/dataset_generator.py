import os
import sys
from faker import Faker
from ollama import generate
import pandas as pd
from llm_data_quality_assistant.llm_models import OllamaModel
from enums import Models

from llm_data_quality_assistant.helper_functions.csv_helper import (
    extract_csv_from_prompt,
)

"""
Dataset generator for generating a dataset using LLMs.
"""


def llm_dataset(
    n_rows: int, column_headers: list[str], model_name=Models.OllamaModels.GEMMA3_4B
) -> pd.DataFrame:
    theme = f"""
    Create data representing students at the techincal university of munich. Create data represents realistic names, grades, ethnicity, ages etc.
    """

    prompt = f"""
    You are a CSV data generator specifically tailored for creating unique datasets that do not exist anywhere on the internet.
    Your primary focus is to output only valid CSV data without any additional text or formatting that could interfere with production processes.
    Your role is to serve as a precise and reliable CSV data creator, ensuring that all entries are original and formatted correctly according to the specified headers.

    The data should NOT look like typical testing data but like actual, realistic data that could fool anyone. The data should look like a regular dataset and not a testing set with basic data and require stringent adherence to CSV formatting standards.

    Your task is to take the column headers: {column_headers}
    as input and generate a dataset with {n_rows} entries, ensuring that all data is unique and valid for production use. The entries must be structured as a CSV string output only, without any explanations or additional content.
    Output format: valid CSV string.
    All of the data entries should follow the theme {theme}.
    Your output will be used for further processing. So DO NOT OUTPUT ANYTHING OTHER THAN THE CSV. Otherwise bad things will happen to the production line, and you dont want this to happen.
    """
    model = OllamaModel(model_name=model_name)
    message = model.generate(prompt, stream=False)
    if type(message) == str:
        return extract_csv_from_prompt(message)
    else:
        return pd.DataFrame()


"""
Dataset generator for generating a simple dataset with no significant meta information.
Data is just simple random data.
"""


def simple_dataset(n_rows: int, n_cols: int) -> pd.DataFrame:
    faker = Faker()

    data = []
    for r in range(n_rows):
        data.append(
            [
                faker.name(),
                faker.date_of_birth(minimum_age=18, maximum_age=80),
                faker.address(),
                faker.country(),
                faker.email(),
                faker.phone_number(),
                faker.job(),
                faker.company(),
            ]
        )

    return pd.DataFrame(
        data,
    )


if __name__ == "__main__":
    print("Generating dataset...")
    column_headers = [
        "name, date_of_birth, address, country, email, phone_number, job, company, average_bachelors_grade on european grade system"
    ]
    print(
        llm_dataset(
            20, column_headers=column_headers, model_name=Models.OllamaModels.GEMMA3_4B
        )
    )
