from os import PathLike

import pandas as pd

from matcher.normalize import extract_size, normalize_brand, normalize_name
from matcher.parsing import parse_item_info, parse_sizing_comp
from matcher.schemas import ProductRecord


def load_catalog_csv(path: str | PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def dataframe_to_products(rows: list[dict[str, str]]) -> list[ProductRecord]:
    products = []
    for row in rows:
        item_id = row.get("item_id", "") or ""
        name = row.get("name", "") or ""
        brand_raw = row.get("brand_raw", "") or ""
        sizing = parse_sizing_comp(row.get("sizing_comp", ""))
        size_value, size_unit = extract_size(
            sizing.get("size_user_friendly", "") or row.get("size_raw", "")
        )
        products.append(
            ProductRecord(
                item_id=item_id,
                name=name,
                brand_raw=brand_raw,
                brand_norm=normalize_brand(brand_raw),
                description=row.get("description", ""),
                category_path=parse_item_info(row.get("item_info", "")),
                size_value=size_value,
                size_unit=size_unit,
                tokens_full=normalize_name(name).split(),
                raw_payload=row,
            )
        )
    return products
