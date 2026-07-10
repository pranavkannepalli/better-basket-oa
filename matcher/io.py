import csv
from os import PathLike
from pathlib import Path

import pandas as pd

from matcher.schemas import MatchDecision

from matcher.normalize import (
    extract_attribute_flags,
    extract_core_tokens,
    extract_form_flags,
    extract_pack_count,
    extract_size,
    normalize_brand,
    normalize_name,
)
from matcher.identifiers import extract_global_ids, extract_source_ids, extract_source_product_id
from matcher.parsing import parse_item_info, parse_sizing_comp
from matcher.schemas import ProductRecord

PRIVATE_LABEL_BRANDS = {
    "better homes & gardens",
    "equate",
    "great value",
    "mainstays",
    "marketside",
    "parents choice",
    "sam's choice",
    "wegmans",
    "wegmans organic",
}


def load_catalog_csv(path: str | PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def _parse_boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _category_path(row: dict[str, str], item_info: dict) -> list[str]:
    path = parse_item_info(row.get("item_info", ""))
    for key in ("department", "category", "subcategory", "item_type"):
        value = row.get(key, "")
        if value and value not in path:
            path.append(value)
    return path


def _is_private_label(row: dict[str, str], brand_norm: str) -> bool:
    tags = normalize_name(row.get("tags", ""))
    return (
        _parse_boolish(row.get("is_private_label", ""))
        or brand_norm in PRIVATE_LABEL_BRANDS
        or "wegmans_brand" in row.get("tags", "")
        or "wegmans brand" in tags
    )


def write_matches_csv(path: Path, decisions: list[MatchDecision]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id_a",
                "item_id_b",
                "confidence",
                "match_quality",
                "decision_source",
                "review_flag",
            ],
        )
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.model_dump())


def write_submission_csv(path: Path, decisions: list[MatchDecision], min_confidence: float = 0.4) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id_A", "item_id_B"])
        writer.writeheader()
        for decision in decisions:
            if decision.confidence < min_confidence:
                continue
            writer.writerow({"item_id_A": decision.item_id_a, "item_id_B": decision.item_id_b})


def dataframe_to_products(rows: list[dict[str, str]]) -> list[ProductRecord]:
    products = []
    for row in rows:
        brand_raw = row.get("brand_raw", "") or ""
        brand_norm = normalize_brand(brand_raw)
        sizing = parse_sizing_comp(row.get("sizing_comp", ""))
        item_info_raw = row.get("item_info", "")
        item_info = parse_sizing_comp(item_info_raw)
        combined_text = " ".join(
            [
                _text(row.get("name", "")),
                _text(row.get("name_clean", "")),
                _text(row.get("description", "")),
                _text(row.get("tags", "")),
                _text(row.get("is_organic", "")),
                _text(item_info.get("storage_type", "")),
                _text(item_info.get("packaging_description", "")),
            ]
        )
        size_text = _text(sizing.get("size_user_friendly", "")) or _text(row.get("size_raw", ""))
        size_value, size_unit = extract_size(
            size_text
        )
        global_ids, global_id_flags = extract_global_ids(row, item_info)
        source_ids, source_id_flags = extract_source_ids(row, item_info)
        url = row.get("url", "")
        products.append(
            ProductRecord(
                item_id=row["item_id"],
                name=row["name"],
                url=url,
                brand_raw=brand_raw,
                brand_norm=brand_norm,
                description=row.get("description", ""),
                category_path=_category_path(row, item_info),
                private_label_flag=_is_private_label(row, brand_norm),
                size_value=size_value,
                size_unit=size_unit,
                pack_count=extract_pack_count(
                    " ".join([_text(row.get("name", "")), size_text, _text(row.get("size_raw", ""))])
                ),
                form_flags=extract_form_flags(combined_text),
                tokens_core=extract_core_tokens(row["name"]),
                tokens_full=normalize_name(row["name"]).split(),
                attribute_flags=extract_attribute_flags(combined_text),
                global_ids=global_ids,
                source_product_id=source_ids.get("url_product_id", extract_source_product_id(url)),
                source_ids=source_ids,
                identifier_flags=sorted(set(global_id_flags + source_id_flags)),
            )
        )
    return products
