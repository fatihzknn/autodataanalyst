import pandas as pd

def load_file(file, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    filename = file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file, sep=sep, encoding=encoding)

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(file)

    raise ValueError("Unsupported file type")
