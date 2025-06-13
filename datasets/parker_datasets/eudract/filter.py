import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from filter_by_key import filter_by_key


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filter_by_key(
        gold_standard_path=os.path.join(
            script_dir, "raw/eudract_gold_standard_pivoted.csv"
        ),
        corrupted_dataset_path=os.path.join(script_dir, "raw/eudract.csv"),
        primary_key="eudract_number",
        n=1000,
        out_gold=os.path.join(script_dir, "eudract_cleaned_gold_first1000.csv"),
        out_corrupt=os.path.join(script_dir, "eudract_corrupted_first1000.csv"),
    )


if __name__ == "__main__":
    main()
