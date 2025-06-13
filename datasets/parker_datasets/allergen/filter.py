import argparse
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from filter_by_key import filter_by_key


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filter_by_key(
        gold_standard_path=os.path.join(
            script_dir, "raw/gold_standard_alergene_pivoted.csv"
        ),
        corrupted_dataset_path=os.path.join(script_dir, "raw/allergen.csv"),
        primary_key="code",
        n=1000,
        out_gold=os.path.join(script_dir, "allergen_cleaned_gold_first1000.csv"),
        out_corrupt=os.path.join(script_dir, "allergen_corrupted_first1000.csv"),
    )


if __name__ == "__main__":
    main()
