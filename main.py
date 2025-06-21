from llm_data_quality_assistant.evaluation import (
    evaluate_dataset_micro,
    evaluate_dataset_macro,
)
from llm_data_quality_assistant.pipeline import Pipeline
import pandas as pd
from pprint import pprint


gold_standard = pd.read_csv(
    "./datasets/parker_datasets/allergen/allergen_cleaned_gold_first1000.csv"
)

corrupted_dataset = pd.read_csv(
    "./datasets/parker_datasets/allergen/allergen_corrupted_first1000.csv"
)

cleaned_dataset = pd.read_csv(
    "./analysis/repaired_llm/allergen/merged_dataset_gemini_2_0_flash_lite_50_rows_context.csv"
)

output = Pipeline.standardize_datasets(
    datasets=[gold_standard, cleaned_dataset, corrupted_dataset],
    primary_key="code",
)
print(output)
gold_standard = output[0]
cleaned_dataset = output[1]
corrupted_dataset = output[2]


micro_stats = evaluate_dataset_micro(
    gold_standard=gold_standard,
    cleaned_dataset=cleaned_dataset,
    corrupted_dataset=corrupted_dataset,
)

pprint(micro_stats)
