import csv
import json
from os import PathLike
from pathlib import Path

import pandas as pd

from matcher.schemas import MatchDecision

from matcher.normalize import (
    ATTRIBUTE_KEYWORDS,
    FORM_KEYWORDS,
    STOPWORDS,
    extract_attribute_flags,
    extract_form_flags,
    extract_pack_count,
    extract_size,
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


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _series_text(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([""] * len(frame), index=frame.index, dtype="string")
    return frame[column].fillna("").astype(str)


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.str.lower()
        .str.replace(r"[^a-z0-9\.\s]", " ", regex=True)
        .str.split()
        .str.join(" ")
    )


def _normalize_brand_series(series: pd.Series) -> pd.Series:
    return series.str.lower().str.strip().str.split().str.join(" ")


def _core_tokens_from_normalized(normalized_name: str) -> list[str]:
    return [
        token
        for token in normalized_name.split()
        if token not in STOPWORDS and token not in ATTRIBUTE_KEYWORDS and token not in FORM_KEYWORDS
    ]


def _category_path(row: dict[str, str], item_info: dict) -> list[str]:
    path = parse_item_info(row.get("item_info", ""))
    for key in ("department", "category", "subcategory", "item_type"):
        value = row.get(key, "")
        if value and value not in path:
            path.append(value)
    return path


PRODUCT_DETAIL_FIELDS = [
    "item_id",
    "name",
    "url",
    "brand_raw",
    "brand_norm",
    "description",
    "category_path",
    "private_label_flag",
    "size_value",
    "size_unit",
    "pack_count",
    "form_flags",
    "attribute_flags",
    "global_ids",
    "source_product_id",
    "source_ids",
    "identifier_flags",
    "tokens_core",
    "tokens_full",
]


def _csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return "" if value is None else value


def _product_detail_row(prefix: str, product: ProductRecord | None) -> dict:
    if product is None:
        return {f"{prefix}_{field}": "" for field in PRODUCT_DETAIL_FIELDS}
    data = product.model_dump()
    return {f"{prefix}_{field}": _csv_value(data.get(field)) for field in PRODUCT_DETAIL_FIELDS}


def _decision_row(decision: MatchDecision) -> dict:
    return {
        "item_id_a": decision.item_id_a,
        "item_id_b": decision.item_id_b,
        "confidence": decision.confidence,
        "match_quality": decision.match_quality,
        "decision_source": decision.decision_source,
        "review_flag": decision.review_flag,
    }


def write_matches_csv(
    path: Path,
    decisions: list[MatchDecision],
    products_a_by_id: dict[str, ProductRecord] | None = None,
    products_b_by_id: dict[str, ProductRecord] | None = None,
) -> None:
    decision_fields = [
        "item_id_a",
        "item_id_b",
        "confidence",
        "match_quality",
        "decision_source",
        "review_flag",
    ]
    detail_fields = []
    if products_a_by_id is not None and products_b_by_id is not None:
        detail_fields = [f"a_{field}" for field in PRODUCT_DETAIL_FIELDS]
        detail_fields += [f"b_{field}" for field in PRODUCT_DETAIL_FIELDS]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=decision_fields + detail_fields)
        writer.writeheader()
        for decision in decisions:
            row = _decision_row(decision)
            if products_a_by_id is not None and products_b_by_id is not None:
                row.update(_product_detail_row("a", products_a_by_id.get(decision.item_id_a)))
                row.update(_product_detail_row("b", products_b_by_id.get(decision.item_id_b)))
            writer.writerow(row)


def append_matches_csv(
    path: Path,
    decisions,
    products_a_by_id: dict[str, ProductRecord] | None = None,
    products_b_by_id: dict[str, ProductRecord] | None = None,
) -> None:
    decision_fields = [
        "item_id_a",
        "item_id_b",
        "confidence",
        "match_quality",
        "decision_source",
        "review_flag",
    ]
    detail_fields = []
    if products_a_by_id is not None and products_b_by_id is not None:
        detail_fields = [f"a_{field}" for field in PRODUCT_DETAIL_FIELDS]
        detail_fields += [f"b_{field}" for field in PRODUCT_DETAIL_FIELDS]

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=decision_fields + detail_fields)
        if needs_header:
            writer.writeheader()
        for decision in decisions:
            row = _decision_row(decision)
            if products_a_by_id is not None and products_b_by_id is not None:
                row.update(_product_detail_row("a", products_a_by_id.get(decision.item_id_a)))
                row.update(_product_detail_row("b", products_b_by_id.get(decision.item_id_b)))
            writer.writerow(row)


def write_submission_csv(path: Path, decisions: list[MatchDecision], min_confidence: float = 0.4) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id_A", "item_id_B"])
        writer.writeheader()
        for decision in decisions:
            if decision.confidence < min_confidence:
                continue
            writer.writerow({"item_id_A": decision.item_id_a, "item_id_B": decision.item_id_b})


def append_submission_csv(path: Path, decisions, min_confidence: float = 0.4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id_A", "item_id_B"])
        if needs_header:
            writer.writeheader()
        for decision in decisions:
            if decision.confidence < min_confidence:
                continue
            writer.writerow({"item_id_A": decision.item_id_a, "item_id_B": decision.item_id_b})


def write_detailed_submission_csv(
    path: Path,
    decisions,
    products_a_by_id: dict[str, ProductRecord],
    products_b_by_id: dict[str, ProductRecord],
    min_confidence: float = 0.4,
) -> None:
    write_matches_csv(
        path,
        (decision for decision in decisions if decision.confidence >= min_confidence),
        products_a_by_id,
        products_b_by_id,
    )


def append_detailed_submission_csv(
    path: Path,
    decisions,
    products_a_by_id: dict[str, ProductRecord],
    products_b_by_id: dict[str, ProductRecord],
    min_confidence: float = 0.4,
) -> None:
    append_matches_csv(
        path,
        (decision for decision in decisions if decision.confidence >= min_confidence),
        products_a_by_id,
        products_b_by_id,
    )


def dataframe_to_products(rows: list[dict[str, str]] | pd.DataFrame) -> list[ProductRecord]:
    if isinstance(rows, pd.DataFrame):
        frame = rows
    else:
        frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return []

    brand_raw_values = _series_text(frame, "brand_raw")
    brand_norm_values = _normalize_brand_series(brand_raw_values)
    name_values = _series_text(frame, "name")
    normalized_name_values = _normalize_text_series(name_values)
    tags_values = _series_text(frame, "tags")
    normalized_tags_values = _normalize_text_series(tags_values)
    private_label_values = (
        _series_text(frame, "is_private_label").str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
        | brand_norm_values.isin(PRIVATE_LABEL_BRANDS)
        | tags_values.str.contains("wegmans_brand", regex=False)
        | normalized_tags_values.str.contains("wegmans brand", regex=False)
    )
    name_clean_values = _series_text(frame, "name_clean")
    description_values = _series_text(frame, "description")
    is_organic_values = _series_text(frame, "is_organic")
    size_raw_values = _series_text(frame, "size_raw")
    combined_base_values = (
        name_values
        + " "
        + name_clean_values
        + " "
        + description_values
        + " "
        + tags_values
        + " "
        + is_organic_values
    )
    tokens_full_values = [value.split() for value in normalized_name_values]
    tokens_core_values = [_core_tokens_from_normalized(value) for value in normalized_name_values]
    brand_raw_list = brand_raw_values.tolist()
    brand_norm_list = brand_norm_values.tolist()
    private_label_list = private_label_values.tolist()
    combined_base_list = combined_base_values.tolist()
    size_raw_list = size_raw_values.tolist()
    columns = list(frame.columns)
    products = []
    for index, values in enumerate(frame.itertuples(index=False, name=None)):
        row = dict(zip(columns, values))
        brand_raw = brand_raw_list[index]
        brand_norm = brand_norm_list[index]
        sizing = parse_sizing_comp(row.get("sizing_comp", ""))
        item_info_raw = row.get("item_info", "")
        item_info = parse_sizing_comp(item_info_raw)
        combined_text = " ".join(
            [
                combined_base_list[index],
                _text(item_info.get("storage_type", "")),
                _text(item_info.get("packaging_description", "")),
            ]
        )
        size_text = _text(sizing.get("size_user_friendly", "")) or size_raw_list[index]
        size_value, size_unit = extract_size(size_text)
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
                private_label_flag=bool(private_label_list[index]),
                size_value=size_value,
                size_unit=size_unit,
                pack_count=extract_pack_count(
                    " ".join([_text(row.get("name", "")), size_text, _text(row.get("size_raw", ""))])
                ),
                form_flags=extract_form_flags(combined_text),
                tokens_core=tokens_core_values[index],
                tokens_full=tokens_full_values[index],
                attribute_flags=extract_attribute_flags(combined_text),
                global_ids=global_ids,
                source_product_id=source_ids.get("url_product_id", extract_source_product_id(url)),
                source_ids=source_ids,
                identifier_flags=sorted(set(global_id_flags + source_id_flags)),
            )
        )
    return products
