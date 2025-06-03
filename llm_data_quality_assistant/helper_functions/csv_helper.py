import pandas as pd
import io
import re


def csv_str_to_dataframe(response_text: str) -> pd.DataFrame:
    return pd.read_csv(
        io.StringIO(response_text), sep=",", header=None, engine="python"
    )


def remove_think_tag(response_text: str):
    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)


def extract_content_inside_csv_tag(response_text: str):
    # Findet alle Codeblöcke, die mit ```csv beginnen und mit ``` enden
    pattern = r"```csv\s+(.*?)```"
    matches = re.findall(pattern, response_text, flags=re.DOTALL)
    print(matches)
    return [match.strip() for match in matches]


def extract_csv_from_prompt(response_text: str):
    # Match from first line with commas till end
    response_text = remove_think_tag(response_text)
    try:
        return csv_str_to_dataframe(response_text)
    except OSError:
        pass

    data = csv_str_to_dataframe(extract_content_inside_csv_tag(response_text)[0])

    return data
