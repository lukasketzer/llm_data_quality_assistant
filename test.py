from llm_data_quality_assistant import llm_integration
from llm_data_quality_assistant.llm_models import get_model
from llm_data_quality_assistant.enums import Models
from llm_data_quality_assistant import pipeline
from llm_data_quality_assistant.corruptor import RowCorruptionTypes, CellCorruptionTypes
from llm_data_quality_assistant.enums import Models
import pandas as pd
from pprint import pprint

import pandas as pd
import numpy as np
from pprint import pprint


gold_standard = pd.read_csv(
    "./datasets/parker_datasets/gold_standard_alergene_pivoted.csv"
).head(
    5
)  # Load a sample of the dataset
p = pipeline.Pipeline(gold_standard)

row_corruption_types = [RowCorruptionTypes.SHUFFLE_COLUMNS]
cell_corruption_types = [CellCorruptionTypes.OUTLIER, CellCorruptionTypes.NULL]
print("Corruptig datasets....")
corrupted_datasets, corrupted_coords = p.generate_corrupted_datasets(
    row_corruption_type=row_corruption_types,
    cell_corruption_type=cell_corruption_types,
    severity=0.12,
    output_size=5,
)
print("Corruptig done.")

# Example: merge the corrupted datasets using LLM
print("Start merging...")
merged_df = p.merge_with_llm(corrupted_datasets, chunk_size=5)
print("Merging done.")
print("Merged DataFrame:")
print(merged_df)

stats = p.evaluate_micro(merged_df, corrupted_coords)
print("Micro evaluation stats:")
pprint(stats)
