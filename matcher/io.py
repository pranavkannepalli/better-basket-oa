import pandas as pd


def load_catalog_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")
