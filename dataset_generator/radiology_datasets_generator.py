import pandas as pd
import random


def make_uid():
    return f"1.2.840.{random.randint(1, 999999)}.{random.randint(1, 999999999)}.{random.randint(1, 9999)}"


modality_info = {
    "CT": {
        "rows_cols": [512, 512],
        "series_pool": [
            "BRAIN_WO_CON",
            "CHEST_PE_PROTOCOL",
            "ABDOMEN_W_CON",
            "SINUS_CORONAL",
            "HEAD_TRAUMA_CT",
        ],
    },
    "MR": {
        "rows_cols": [256, 256],
        "series_pool": [
            "BRAIN_T1_AX",
            "SPINE_LUMBAR_SAG_T2",
            "KNEE_PD_FSE",
            "ANKE_STIR_COR",
            "HIP_AX_T1",
        ],
    },
    "XR": {
        "rows_cols": [2048, 2048],
        "series_pool": [
            "CHEST_PA",
            "HAND_AP",
            "ABDOMEN_KUB",
            "FOOT_LATERAL",
            "PELVIS_AP",
        ],
    },
    "US": {
        "rows_cols": [1024, 1024],
        "series_pool": [
            "ABDOMEN_ULTRASOUND",
            "PELVIS_ULTRASOUND",
            "CAROTID_DOPPLER",
            "THYROID_US",
            "RENAL_ULTRASOUND",
        ],
    },
}

entries = []
modalities = list(modality_info.keys())

for _ in range(100):
    modality = random.choice(modalities)
    info = modality_info[modality]
    rows, cols = info["rows_cols"]
    series_desc = random.choice(info["series_pool"])
    entries.append(
        {
            "dicom_uid": make_uid(),
            "rows": rows,
            "columns": cols,
            "series_desc": series_desc,
            "modality": modality,
        }
    )

df = pd.DataFrame(entries)

print("Radiology modality dataset (100 rows)")
print(df)

exit(1)

file_path = "/mnt/data/radiology_modality_dataset.csv"
df.to_csv(file_path, index=False)
