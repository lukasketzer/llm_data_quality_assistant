# LLM-Based Data Quality Assistant

## Overview

This project provides a modular framework for generating, corrupting, cleaning, and evaluating tabular datasets using Large Language Models (LLMs). It is designed for research and practical experimentation with data quality, leveraging LLMs for data cleaning and merging tasks.

## Features

- **Dataset Generation**: Create realistic synthetic datasets using LLMs or Faker.
- **Corruption Module**: Apply a variety of row and cell-level corruptions to simulate real-world data quality issues.
- **LLM Integration**: Merge and clean datasets using LLMs (Ollama, Gemini, OpenAI supported).
- **Evaluation**: Quantitatively evaluate cleaning/merging results using micro and macro metrics.
- **Jupyter Notebooks**: Example notebooks for end-to-end experiments.

## Directory Structure

- `llm_data_quality_assistant/` – Core package  
  - `corruptor.py`, `corruption_functions.py`: Corruption logic and types  
  - `dataset_generator.py`: Synthetic dataset generation  
  - `llm_integration.py`, `llm_models.py`: LLM-based merging/cleaning  
  - `evaluation.py`: Metrics and evaluation  
  - `pipeline.py`: High-level workflow API  
  - `enums/Models.py`: Model enums  
- `datasets/` – Example datasets (gold standards, corrupted, public)
- `analysis/` – Jupyter notebooks for experiments
- `requirements.txt` – Python dependencies

## Installation

1. Clone the repository.
2. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. (Optional) Set up API keys for Gemini/OpenAI in your environment if using those models.

## Usage

### Example: End-to-End Pipeline in Python

```python
from llm_data_quality_assistant import pipeline
from llm_data_quality_assistant.corruptor import RowCorruptionTypes, CellCorruptionTypes
from llm_data_quality_assistant.enums import Models
import pandas as pd

# Load a gold standard dataset
gold_standard = pd.read_csv('datasets/parker_datasets/flight/flight_gold_standard_pivoted.csv')
p = pipeline.Pipeline(gold_standard)

# Generate corrupted datasets
corrupted_datasets, corrupted_coords = p.generate_corrupted_datasets(
    row_corruption_type=[RowCorruptionTypes.SHUFFLE_COLUMNS],
    cell_corruption_type=[CellCorruptionTypes.TYPO],
    severity=0.1,
    output_size=3
)

# Clean/merge with LLM
merged_df = p.merge_with_llm(corrupted_datasets, model_name=Models.GeminiModels.GEMINI_2_0_FLASH)

# Evaluate
results = p.evaluate_micro(merged_df, corrupted_coords)
print(results)
```

### Jupyter Notebooks

See `analysis/flight_llm.ipynb` and `analysis/allergen_llm.ipynb` for full experiment workflows.

## Supported Corruption Types

- **Row-level**: swap, delete, shuffle columns, reverse
- **Cell-level**: outlier, null, incorrect datatype, inconsistent format, swap cells, case error, truncate, rounding error, encoding error, typo

## Supported LLMs

- Ollama (local)
- Gemini (Google)
- OpenAI (API, not implemented)

## Datasets

- `datasets/parker_datasets/` – Gold standards and corrupted versions for flight, allergen, eudract
- `datasets/public_dataset/` – Example public datasets
- `datasets/selfwritte_dataset/`, `datasets/simple_dataset/` – Synthetic examples

## Requirements

See `requirements.txt` for all dependencies (`pandas`, `numpy`, `ollama`, `google-genai`, `faker`, etc).

## License

For academic use only. See LICENSE file if present.
