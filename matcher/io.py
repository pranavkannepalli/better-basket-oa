from os import PathLike

import pandas as pd


def load_catalog_csv(path: str | PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")
