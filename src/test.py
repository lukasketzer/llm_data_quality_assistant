import pandas as pd
import numpy as np

if __name__ == "__main__":
    gold_standard_alerene = pd.read_csv(
        "datasets/parker_datasets/allergen_gold_standard.csv"
    )
    gold_standard_alerene = gold_standard_alerene.pivot(
        index="tid", columns="attribute", values="correct_val"
    )

    # Ensure consistent column order (matching your sample output)
    desired_columns = [
        "code",
        "nuts",
        "almondnuts",
        "brazil_nuts",
        "macadamia_nuts",
        "hazelnut",
        "pistachio",
        "walnut",
        "cashew",
        "celery",
        "crustaceans",
        "eggs",
        "fish",
        "gluten",
        "lupin",
        "milk",
        "molluscs",
        "mustard",
        "peanut",
        "sesame",
        "soy",
        "sulfite",
    ]
    gold_standard_alerene = gold_standard_alerene.reindex(
        columns=desired_columns
    ).fillna(0)

    gold_standard_alerene.to_csv("gold_standard_alergene_pivoted.csv")

    alergen_raw = pd.read_csv(
        "datasets/parker_datasets/allergen.csv",
    )

    cods_raw: np.ndarray = alergen_raw["code"].unique()
    print(f"Number of unique codes in raw data: {len(cods_raw)}")
    codes_gold_standard: np.ndarray = gold_standard_alerene["code"].unique()
    print(f"number of unique codes in gold standard: {len(codes_gold_standard)}")

    # Count occurrences of shared elements
    common_elements = np.intersect1d(cods_raw, codes_gold_standard)
    count = sum(
        min(np.count_nonzero(cods_raw == x), np.count_nonzero(codes_gold_standard == x))
        for x in common_elements
    )
    print(f"counts: {count} shared elements between raw and gold standard datasets")
    pass
