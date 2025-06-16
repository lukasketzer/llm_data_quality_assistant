import sys
import os
import time
import uuid

from pydantic import RootModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import json
from llm_data_quality_assistant.llm_integration import get_model
from llm_data_quality_assistant.enums import Models
from llm_data_quality_assistant.llm_integration import __generate_pydantic_structure
from tqdm import tqdm  # Add tqdm for status bar

df = pd.read_csv("./datasets/self_generated_dataset/Radiology_modality_sample.csv")

struct = __generate_pydantic_structure(dataset=df)
model = get_model(Models.GeminiModels.GEMINI_2_0_FLASH_LITE)


class ListStruct(RootModel[list[struct]]):
    pass


struct = ListStruct

rows = []
rpm = 30
min_interval = 60 / rpm if rpm > 0 else 0
last_request_time = None


for _ in tqdm(range(100), desc="Generating rows"):  # Add tqdm status bar
    now = time.time()
    if last_request_time is not None and min_interval > 0:
        elapsed = now - last_request_time
        to_wait = min_interval - elapsed
        if to_wait > 0:
            time.sleep(to_wait)

    last_request_time = time.time()

    message = model.generate(
        prompt=f"Generate 1 row of random data based on the provided dataset structure",
        format=struct,
    )

    data = json.loads(message)
    output = pd.DataFrame(data)
    rows.append(output)

rows = pd.concat(rows, ignore_index=True)

print(rows)
# Generate new unique UIDs for the "dicom_uid" column
rows["dicom_uid"] = [str(uuid.uuid4()) for _ in range(len(rows))]
rows.to_csv("./datasets/llm_dataset/Radiology_modality_sample_output.csv", index=False)
