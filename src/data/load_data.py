from pathlib import Path
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    data_path = Path(path)

    if data_path.suffix.lower() == ".csv":
        return pd.read_csv(data_path)

    if data_path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(data_path)

    raise ValueError(f"Unsupported data file type: {data_path.suffix}")
