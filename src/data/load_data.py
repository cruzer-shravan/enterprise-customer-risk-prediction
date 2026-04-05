# Data loading -->  src/data/load_data.py 

import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)