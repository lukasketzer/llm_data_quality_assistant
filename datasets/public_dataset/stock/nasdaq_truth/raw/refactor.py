import pandas as pd
import os

columns = [
    "Symbol",
    "Change %",
    "Last trading price",
    "Open price",
    "Change $",
    "Volume",
    "Today's high",
    "Today's low",
    "Previous close",
    "52wk High",
    "52wk Low",
    "Shares Outstanding",
    "P/E",
    "Market cap",
    "Yield",
    "Dividend",
    "EPS",
    "Empty",
]

print(len(columns))


# List all CSV files in the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

csv_files = [file for file in os.listdir(current_dir) if file.endswith(".txt")]
print(csv_files)

# Read each CSV file into a DataFrame and print the first few rows
for file in csv_files:
    print(f"Reading {file}...")
    df = pd.read_csv(
        f"{current_dir}/{file}",
        sep="\t",
        header=None,
        names=columns,
        engine="python",
    )

    # Drop the last column ("Empty")
    df = df.iloc[:, :-1]

    df.to_csv(f"{current_dir}/{file.replace('.txt', '.csv')}", index=False)
    print(df.head())
