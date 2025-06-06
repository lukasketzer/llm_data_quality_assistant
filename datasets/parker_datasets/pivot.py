import sys
import pandas as pd


def pivot_csv(input_path, output_path):
    df = pd.read_csv(input_path)
    # Pivot the table: index='tid', columns='attribute', values='correct_val'
    df_pivoted = df.pivot(
        index="tid", columns="attribute", values="correct_val"
    ).reset_index()
    # Save to CSV
    df_pivoted.drop(columns=["tid"], inplace=True)
    df_pivoted.to_csv(output_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(f"Usage: python {sys.argv[0]} <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = (
        sys.argv[2]
        if len(sys.argv) > 2
        else input_csv[:-4] + "_pivoted" + input_csv[-4:]
    )
    pivot_csv(input_csv, output_csv)
