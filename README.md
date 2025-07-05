# LLM Data Quality Assistant

A Python framework for data quality assessment, corruption simulation, and repair using Large Language Models (LLMs).

## Project Overview

This project provides a comprehensive solution for:

1. **Data Corruption Simulation**: Generate realistic data quality issues in datasets
2. **LLM-based Data Repair**: Leverage various LLM models to detect and fix corrupted data
3. **Evaluation Framework**: Compare performance of different repair approaches
4. **Pipeline Architecture**: End-to-end workflow from corruption to repair and evaluation

The project focuses on evaluating the effectiveness of LLMs in data cleaning tasks compared to traditional approaches.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llm_data_quality_assistant

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for API keys (create a .env file)
touch .env
# Add your API keys to the .env file:
# OPENAI_API_KEY=your_openai_key
# GEMINI_API_KEY=your_gemini_key
```

## Project Structure

- **`llm_data_quality_assistant/`**: Core library implementing the data quality framework
  - `corruptor.py`: Functions to inject realistic data corruptions
  - `llm_integration.py`: Interface to various LLM models
  - `pipeline.py`: End-to-end data processing pipeline
  - `llm_models.py`: Implementations for different LLM providers (OpenAI, Gemini, Ollama)
  - `enums/`: Enumeration types for corruption types and models

- **`datasets/`**: Contains original and corrupted datasets
  - `self_generated_dataset/`: Custom datasets
  - `parker_datasets/`: Datasets from the Parker framework
  - `llm_dataset/`: LLM-generated datasets

- **`analysis/`**: Jupyter notebooks for experiments and analysis
  - Notebooks for different datasets (allergen, eudract, flight, etc.)
  - `evaluation.py`: Metrics calculation and evaluation utilities

- **`dataset_generator/`**: Utilities to generate test datasets
  - `llm_dataset_generator.py`: Generate datasets using LLMs
  - `radiology_datasets_generator.py`: Generate specialized radiology datasets

- **`tests/`**: Unit tests for the framework components

## Usage Examples

### Basic Usage

```python
import pandas as pd
from llm_data_quality_assistant.pipeline import Pipeline
from llm_data_quality_assistant.enums.CorruptionTypes import CellCorruptionTypes, RowCorruptionTypes
from llm_data_quality_assistant.enums import Models
from analysis.evaluation import evaluate_dataset_micro

# Load a dataset
df = pd.read_csv("datasets/self_generated_dataset/Radiology_modality_sample.csv")

# Create a pipeline and corrupt the dataset
cell_corruption_types = [CellCorruptionTypes.OUTLIER, CellCorruptionTypes.NULL]
row_corruption_types = [RowCorruptionTypes.SWAP_ROWS]
corrupted_dfs = Pipeline.generate_corrupted_datasets(
    dataset=df,
    cell_corruption_types=cell_corruption_types,
    row_corruption_types=row_corruption_types,
    columns_to_exclude=["dicom_uid"],  # Exclude primary key
    severity=0.1,  # 10% corruption
    output_size=1
)
corrupted_df = corrupted_dfs[0]  # Get the first corrupted dataset

# Repair with LLM using the Pipeline
repaired_df = Pipeline.merge_with_llm(
    dataset=corrupted_df,
    primary_key="dicom_uid",
    model_name=Models.OpenAIModels.GPT_4_1_MINI  # Choose from OpenAIModels, GeminiModels, or OllamaModels
)

# Evaluate results
results = evaluate_dataset_micro(
    gold_standard=df,              # Original clean dataset as ground truth
    cleaned_dataset=repaired_df,   # LLM-repaired dataset
    original_dataset=corrupted_df, # Corrupted dataset before repair
    primary_key="dicom_uid"        # Primary key for dataset alignment
)
print(f"Repair precision: {results['precision']}")
print(f"Repair recall: {results['recall']}")
print(f"F1 score: {results['f1']}")
```

### Running Experiments

The project includes several Jupyter notebooks for running experiments with different datasets:

1. Open and run the Jupyter notebooks in the `analysis/` directory:

   ```bash
   jupyter lab
   ```

2. Navigate to notebooks like `allergen_llm.ipynb`, `eudract_option1.ipynb`, etc.

### Using Different LLM Models

The framework supports multiple LLM providers. Here's how to use them:

```python
from llm_data_quality_assistant.pipeline import Pipeline
from llm_data_quality_assistant.enums import Models

# Using OpenAI models
repaired_df_openai = Pipeline.merge_with_llm(
    dataset=corrupted_df,
    primary_key="dicom_uid",
    model_name=Models.OpenAIModels.GPT_4_1_MINI
)

# Using Google Gemini models
repaired_df_gemini = Pipeline.merge_with_llm(
    dataset=corrupted_df,
    primary_key="dicom_uid",
    model_name=Models.GeminiModels.GEMINI_2_0_FLASH_LITE
)

# Using Ollama (local) models
repaired_df_ollama = Pipeline.merge_with_llm(
    dataset=corrupted_df,
    primary_key="dicom_uid",
    model_name=Models.OllamaModels.LLAMA3_LATEST
)
```

#### Advanced Configuration Options

The `merge_with_llm` method supports several parameters for fine-tuning the repair process:

```python
repaired_df = Pipeline.merge_with_llm(
    dataset=corrupted_df,
    primary_key="dicom_uid",
    model_name=Models.GeminiModels.GEMINI_1_5_FLASH,
    rpm=30,              # Limit to 30 requests per minute (0 for no limit)
    additional_prompt="Ensure all dates are in YYYY-MM-DD format and all numerical values are valid.",
    verbose=True,        # Show detailed logging of LLM prompts and responses
    status_bar=True,     # Show progress bar during processing
    strict=True          # Enforce strict validation of LLM responses
)
```

**Parameter Descriptions:**

- **dataset**: The corrupted dataset to be repaired
- **primary_key**: Column name to use as the primary key for merging
- **model_name**: The LLM model to use (from OpenAIModels, GeminiModels, or OllamaModels)
- **rpm**: Rate limiting (requests per minute), set to 0 for no limit
- **additional_prompt**: Custom instructions to guide the LLM's repair process
- **verbose**: When True, prints detailed information about each repair operation
- **status_bar**: When True, shows a progress bar during processing
- **strict**: When True, enforces strict validation of LLM responses

**Examples for Specific Use Cases:**

```python
# Example 1: Quick repair with minimal logging
quick_repair = Pipeline.merge_with_llm(
    dataset=corrupted_df,
    primary_key="dicom_uid",
    model_name=Models.OpenAIModels.GPT_4_1_NANO,  # Fastest/cheapest model
    verbose=False,
    status_bar=True
)

# Example 2: Repair with domain-specific guidance
medical_repair = Pipeline.merge_with_llm(
    dataset=medical_df,
    primary_key="patient_id",
    model_name=Models.GeminiModels.GEMINI_2_0_FLASH,
    additional_prompt="""
        - Ensure all medical codes follow ICD-10 format
        - Blood pressure values should be in format 'systolic/diastolic' (e.g., '120/80')
        - Dates should be in YYYY-MM-DD format
        - All lab values should be numeric with appropriate units
    """,
    strict=True  # Enforce strict validation
)

# Example 3: Offline repair using local Ollama models
offline_repair = Pipeline.merge_with_llm(
    dataset=corrupted_df,
    primary_key="id",
    model_name=Models.OllamaModels.LLAMA3_LATEST,
    verbose=True  # Show detailed logs to monitor local model performance
)
```

## Supported Corruption Types

### Row-level Corruptions

- `SWAP_ROWS`: Swap rows within the dataset
- `DELETE_ROWS`: Remove rows from the dataset
- `SHUFFLE_COLUMNS`: Shuffle column values within rows
- `REVERSE_ROWS`: Reverse the order of rows

### Cell-level Corruptions

- `OUTLIER`: Replace values with outliers
- `NULL`: Replace values with NULL
- `INCORRECT_DATATYPE`: Change the datatype of values
- `SWAP_CELLS`: Swap cell values within columns
- `CASE_ERROR`: Modify string case (upper/lower/title)
- `TRUNCATE`: Truncate string values
- `ROUNDING_ERROR`: Round numeric values incorrectly
- `TYPO`: Introduce typographical errors in strings

## Supported LLM Models

### OpenAI Models

- `GPT_4_1_NANO`: gpt-4.1-nano-2025-04-14 (cheapest)
- `GPT_4_1_MINI`: gpt-4.1-mini-2025-04-14
- `GPT_4_1`: gpt-4.1-2025-04-14
- `O4_MINI`: o4-mini-2025-04-16
- `O3_MINI`: o3-mini-2025-01-31

### Google Gemini Models

- `GEMINI_2_0_FLASH`: gemini-2.0-flash
- `GEMINI_2_0_FLASH_LITE`: gemini-2.0-flash-lite
- `GEMINI_2_5_FLASH_LITE_PREVIEW_06_17`: gemini-2.5-flash-lite-preview-06-17
- `GEMINI_1_5_PRO`: gemini-1.5-pro
- `GEMINI_1_5_FLASH`: gemini-1.5-flash
- `GEMINI_PRO`: gemini-pro
- `GEMINI_ULTRA`: gemini-ultra
- `GEMMA_3_1B`: gemma-3-1b-it
- `GEMMA_3_12B`: gemma-3-12b-it

### Ollama Models

- `GEMMA3_4B`: gemma3:4b
- `GEMMA3_12B`: gemma3:12b
- `GEMMA3_1B`: gemma3:1b
- `DEEPSEEK_R1_7B`: deepseek-r1:7b
- `DEEPSEEK_R1_1_5B`: deepseek-r1:1.5b
- `DEEPSEEK_R1_LATEST`: deepseek-r1:latest
- `QWEN3_14B`: qwen3:14b
- `QWEN3_4B`: qwen3:4b
- `QWEN3_1_7B`: qwen3:1.7b
- `QWEN3_LATEST`: qwen3:latest
- `LLAMA3_LATEST`: llama3:latest

## Testing

Run the test suite to verify the framework components:

```bash
pytest tests/
```
