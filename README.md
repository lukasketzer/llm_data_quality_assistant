# LLM-Based Data Quality Assistant

## Overview

This project provides a modular, research-oriented framework for generating, corrupting, cleaning, and evaluating tabular datasets using Large Language Models (LLMs). It is designed for experimentation with data quality, leveraging LLMs for data cleaning and merging tasks.

## Features

- **Dataset Generation**: Create realistic synthetic datasets using LLMs or Faker.
- **Corruption Module**: Apply a variety of row and cell-level corruptions to simulate real-world data quality issues.
- **LLM Integration**: Merge and clean datasets using LLMs (Ollama, Gemini, OpenAI supported).
- **Evaluation**: Quantitatively evaluate cleaning/merging results using micro and macro metrics.
- **Jupyter Notebooks**: Example notebooks for end-to-end experiments.
- **Extensive Testing**: Pytest-based test suite for core modules.

## Directory Structure

- `llm_data_quality_assistant/` – Core package  
  - `corruptor.py`, `corruption_functions.py`: Corruption logic and types  
  - `evaluation.py`: Metrics and evaluation  
  - `llm_integration.py`, `llm_models.py`: LLM-based merging/cleaning  
  - `merge_baseline.py`: Baseline merging logic  
  - `pipeline.py`: High-level workflow API  
  - `enums/`: Model and corruption type enums
- `dataset_generator/` – Synthetic dataset generation scripts
- `datasets/` – Example datasets (gold standards, corrupted, public)
  - `llm_dataset/`, `parker_datasets/`, `public_dataset/`, `simple_dataset/`
- `analysis/` – Jupyter notebooks for experiments
- `tests/` – Pytest-based unit tests
- `requirements.txt` – Python dependencies

## Installation

1. Clone the repository.
2. (Recommended) Create a virtual environment:

   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. (Optional) Set up API keys for Gemini/OpenAI in your environment if using those models. For example:
   - `GOOGLE_API_KEY` for Gemini
   - `OPENAI_API_KEY` for OpenAI (not implemented)

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

See `analysis/flight_llm.ipynb` and `analysis/allergen_llm.ipynb` for full experiment workflows. To run a notebook:

```sh
jupyter notebook analysis/flight_llm.ipynb
```

## Supported Corruption Types

- **Row-level**: swap, delete, shuffle columns, reverse
- **Cell-level**: outlier, null, incorrect datatype, inconsistent format, swap cells, case error, truncate, rounding error, encoding error, typo

## Supported LLMs

- Ollama (local)
- Gemini (Google)
- OpenAI (API, not implemented)

## Datasets

- `datasets/parker_datasets/` – Gold standards and corrupted versions for flight, allergen, eudract
- `datasets/public_dataset/` – Example public datasets (e.g., stock data)
- `datasets/llm_dataset/`, `datasets/simple_dataset/` – Synthetic and simple examples

## Testing

Run the test suite with:

```sh
pytest tests/
```

## Requirements

All dependencies are listed in `requirements.txt`. Key packages include:
- pandas, numpy, python-dotenv, pytest, Faker, requests
- ollama, google-genai, openai
- jupyter, notebook, jupyterlab, ipywidgets, matplotlib, seaborn
- pyyaml, python-dateutil, pytz, tqdm, python-json-logger

To install all requirements:

```sh
pip install -r requirements.txt
```

