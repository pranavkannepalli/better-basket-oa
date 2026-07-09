# Grocery Product Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that matches every store A item to exactly one best store B item, uses GPT-5 nano heavily for final semantic judgment, emits logs and summary stats, and saves reusable caches.

**Architecture:** Build a small Python package with separate modules for schema parsing, normalization, retrieval, deterministic scoring, GPT scoring, match resolution, persistence, and CLI orchestration. Use local retrieval to avoid a full cross join, then send only a broad finalist set into GPT-5 nano and cache pairwise judgments for reruns.

**Tech Stack:** Python 3.11+, pandas, numpy, rapidfuzz, scikit-learn, openai, pydantic, diskcache, orjson, pytest

---

## File Structure

Planned files and responsibilities:

- Create: `pyproject.toml`
  - Python project metadata and dependencies
- Create: `.env.example`
  - required environment variables for OpenAI
- Create: `matcher/__init__.py`
  - package marker
- Create: `matcher/config.py`
  - runtime settings and paths
- Create: `matcher/schemas.py`
  - typed product, candidate, score, and match models
- Create: `matcher/io.py`
  - CSV loading and output writing
- Create: `matcher/parsing.py`
  - JSON field parsing and raw attribute extraction
- Create: `matcher/normalize.py`
  - title, size, brand, category, and attribute normalization
- Create: `matcher/retrieval.py`
  - broad candidate retrieval across all B rows
- Create: `matcher/scoring.py`
  - deterministic pairwise feature scoring
- Create: `matcher/llm.py`
  - GPT-5 nano client, prompt building, cache lookup, structured parse
- Create: `matcher/resolution.py`
  - final one-best-match selection and quality labeling
- Create: `matcher/persistence.py`
  - cache helpers and JSONL logging
- Create: `matcher/pipeline.py`
  - end-to-end orchestration
- Create: `matcher/cli.py`
  - command-line entrypoint
- Create: `tests/test_parsing.py`
  - parsing coverage
- Create: `tests/test_normalize.py`
  - normalization coverage
- Create: `tests/test_retrieval.py`
  - candidate retrieval coverage
- Create: `tests/test_scoring.py`
  - deterministic scoring coverage
- Create: `tests/test_resolution.py`
  - final selection coverage
- Create: `tests/test_pipeline_smoke.py`
  - end-to-end small-sample smoke test

Repo note:

- The current workspace is not initialized as a git repository.
- If implementation begins here, either run `git init` first or skip commit steps.

### Task 1: Bootstrap The Project

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `matcher/__init__.py`

- [ ] **Step 1: Write the failing environment test**

```python
# tests/test_pipeline_smoke.py
from importlib.util import find_spec


def test_required_packages_are_importable():
    assert find_spec("matcher") is not None
    assert find_spec("rapidfuzz") is not None
    assert find_spec("sklearn") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_smoke.py::test_required_packages_are_importable -v`
Expected: FAIL with `ModuleNotFoundError` or `find_spec("matcher") is None`

- [ ] **Step 3: Write minimal project bootstrap**

```toml
# pyproject.toml
[project]
name = "better-basket-matcher"
version = "0.1.0"
description = "Store A to Store B product matching pipeline"
requires-python = ">=3.11"
dependencies = [
  "diskcache>=5.6.3",
  "numpy>=2.0.0",
  "openai>=1.35.0",
  "orjson>=3.10.0",
  "pandas>=2.2.0",
  "pydantic>=2.7.0",
  "pytest>=8.2.0",
  "python-dotenv>=1.0.1",
  "rapidfuzz>=3.9.0",
  "scikit-learn>=1.5.0",
  "tenacity>=8.4.0",
]

[project.scripts]
bb-match = "matcher.cli:main"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"
```

```bash
# .env.example
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-nano
```

```python
# matcher/__init__.py
__all__ = ["__version__"]
__version__ = "0.1.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pip install -e .`
Expected: editable install succeeds

Run: `pytest tests/test_pipeline_smoke.py::test_required_packages_are_importable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example matcher/__init__.py tests/test_pipeline_smoke.py
git commit -m "chore: bootstrap matcher project"
```

### Task 2: Define Runtime Config And Core Schemas

**Files:**
- Create: `matcher/config.py`
- Create: `matcher/schemas.py`
- Test: `tests/test_parsing.py`

- [ ] **Step 1: Write the failing schema test**

```python
# tests/test_parsing.py
from matcher.schemas import ProductRecord, MatchDecision


def test_product_record_defaults():
    product = ProductRecord(item_id="1", name="Tomato Sauce")
    assert product.brand_norm == ""
    assert product.private_label_flag is False


def test_match_decision_requires_ids():
    decision = MatchDecision(
        item_id_a="1",
        item_id_b="2",
        confidence=0.75,
        match_quality="medium",
        decision_source="llm",
        review_flag=False,
    )
    assert decision.item_id_a == "1"
    assert decision.item_id_b == "2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parsing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'matcher.schemas'`

- [ ] **Step 3: Write minimal config and schema models**

```python
# matcher/config.py
from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    input_a_path: Path = Path("grocery_store_a_items_final.csv")
    input_b_path: Path = Path("grocery_store_b_items_final.csv")
    output_dir: Path = Path("artifacts")
    cache_dir: Path = Path("artifacts/cache")
    llm_model: str = "gpt-5-nano"
    retrieval_k: int = 200
    llm_top_n: int = 24
```

```python
# matcher/schemas.py
from typing import Any, Literal
from pydantic import BaseModel, Field


class ProductRecord(BaseModel):
    item_id: str
    name: str
    brand_raw: str = ""
    brand_norm: str = ""
    description: str = ""
    category_path: list[str] = Field(default_factory=list)
    private_label_flag: bool = False
    size_value: float | None = None
    size_unit: str = ""
    pack_count: int | None = None
    form_flags: list[str] = Field(default_factory=list)
    attribute_flags: list[str] = Field(default_factory=list)
    tokens_core: list[str] = Field(default_factory=list)
    tokens_full: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    item_id_a: str
    item_id_b: str
    deterministic_score: float
    llm_score: float | None = None
    combined_score: float | None = None
    contradiction_flags: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class MatchDecision(BaseModel):
    item_id_a: str
    item_id_b: str
    confidence: float
    match_quality: Literal["high", "medium", "low"]
    decision_source: Literal["rules", "llm", "fallback"]
    review_flag: bool
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parsing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/config.py matcher/schemas.py tests/test_parsing.py
git commit -m "feat: add matcher config and schemas"
```

### Task 3: Build CSV Loading And Raw Field Parsing

**Files:**
- Create: `matcher/io.py`
- Create: `matcher/parsing.py`
- Modify: `matcher/schemas.py`
- Test: `tests/test_parsing.py`

- [ ] **Step 1: Write the failing parsing test**

```python
# tests/test_parsing.py
from matcher.parsing import parse_item_info, parse_sizing_comp


def test_parse_item_info_extracts_category_path():
    raw = '{"category_0":"Grocery","category_1":"Sauces","category_2":"Tomato Sauce"}'
    assert parse_item_info(raw) == ["Grocery", "Sauces", "Tomato Sauce"]


def test_parse_sizing_comp_extracts_user_friendly_size():
    raw = '{"size_user_friendly":"16 fl. oz.","billed_by_weight":false}'
    parsed = parse_sizing_comp(raw)
    assert parsed["size_user_friendly"] == "16 fl. oz."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parsing.py::test_parse_item_info_extracts_category_path -v`
Expected: FAIL with `ImportError` for `matcher.parsing`

- [ ] **Step 3: Write minimal parsing and IO code**

```python
# matcher/parsing.py
import json


def _safe_json_loads(raw: str):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def parse_item_info(raw: str) -> list[str]:
    data = _safe_json_loads(raw)
    return [value for key, value in sorted(data.items()) if key.startswith("category_") and value]


def parse_sizing_comp(raw: str) -> dict:
    data = _safe_json_loads(raw)
    return data if isinstance(data, dict) else {}
```

```python
# matcher/io.py
import pandas as pd


def load_catalog_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_parsing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/io.py matcher/parsing.py tests/test_parsing.py
git commit -m "feat: add raw catalog parsing"
```

### Task 4: Implement Product Normalization

**Files:**
- Create: `matcher/normalize.py`
- Modify: `matcher/parsing.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing normalization tests**

```python
# tests/test_normalize.py
from matcher.normalize import normalize_brand, normalize_name, extract_size


def test_normalize_brand_lowercases_and_trims():
    assert normalize_brand("  Great Value ") == "great value"


def test_normalize_name_strips_noise():
    assert normalize_name("Chobani Whole Milk Greek Yogurt, Honey Blended 5.3 oz") == "chobani whole milk greek yogurt honey blended 5.3 oz"


def test_extract_size_reads_float_and_unit():
    assert extract_size("16 fl. oz.") == (16.0, "fl oz")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalize.py -v`
Expected: FAIL with `ImportError` for `matcher.normalize`

- [ ] **Step 3: Write minimal normalization code**

```python
# matcher/normalize.py
import re


UNIT_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fl\.?\s*oz|oz|lb|ct)")


def normalize_brand(value: str) -> str:
    return " ".join(value.lower().strip().split())


def normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\.\s]", " ", value)
    return " ".join(value.split())


def extract_size(value: str) -> tuple[float | None, str]:
    match = UNIT_PATTERN.search(value.lower())
    if not match:
        return None, ""
    unit = match.group("unit").replace(".", "")
    unit = " ".join(unit.split())
    return float(match.group("value")), unit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/normalize.py tests/test_normalize.py
git commit -m "feat: add core normalization helpers"
```

### Task 5: Convert Raw Rows Into Typed Product Records

**Files:**
- Modify: `matcher/io.py`
- Modify: `matcher/normalize.py`
- Modify: `matcher/parsing.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing typed-record test**

```python
# tests/test_normalize.py
from matcher.io import dataframe_to_products


def test_dataframe_to_products_builds_product_record():
    rows = [
        {
            "item_id": "10",
            "name": "Wegmans Dressing, Basil Vinaigrette",
            "brand_raw": "Wegmans",
            "description": "Gluten free.",
            "item_info": '{"category_0":"Grocery","category_1":"Salad Dressing"}',
            "sizing_comp": '{"size_user_friendly":"16 fl. oz."}',
            "tags": "['_internal_any_gluten_free']",
        }
    ]
    products = dataframe_to_products(rows)
    assert products[0].brand_norm == "wegmans"
    assert products[0].category_path == ["Grocery", "Salad Dressing"]
    assert products[0].size_value == 16.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalize.py::test_dataframe_to_products_builds_product_record -v`
Expected: FAIL with `ImportError` or missing function

- [ ] **Step 3: Write minimal record-building code**

```python
# matcher/io.py
from matcher.normalize import extract_size, normalize_brand, normalize_name
from matcher.parsing import parse_item_info, parse_sizing_comp
from matcher.schemas import ProductRecord


def dataframe_to_products(rows: list[dict[str, str]]) -> list[ProductRecord]:
    products = []
    for row in rows:
        sizing = parse_sizing_comp(row.get("sizing_comp", ""))
        size_value, size_unit = extract_size(sizing.get("size_user_friendly", "") or row.get("size_raw", ""))
        products.append(
            ProductRecord(
                item_id=row["item_id"],
                name=row["name"],
                brand_raw=row.get("brand_raw", ""),
                brand_norm=normalize_brand(row.get("brand_raw", "")),
                description=row.get("description", ""),
                category_path=parse_item_info(row.get("item_info", "")),
                size_value=size_value,
                size_unit=size_unit,
                tokens_full=normalize_name(row["name"]).split(),
                raw_payload=row,
            )
        )
    return products
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalize.py::test_dataframe_to_products_builds_product_record -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/io.py tests/test_normalize.py
git commit -m "feat: build typed product records from raw rows"
```

### Task 6: Add Rich Attribute Extraction

**Files:**
- Modify: `matcher/normalize.py`
- Modify: `matcher/io.py`
- Test: `tests/test_normalize.py`

- [ ] **Step 1: Write the failing attribute extraction test**

```python
# tests/test_normalize.py
from matcher.io import dataframe_to_products


def test_dataframe_to_products_extracts_attributes():
    rows = [
        {
            "item_id": "11",
            "name": "Great Value Organic Tomato Sauce 8 oz",
            "brand_raw": "Great Value",
            "description": "Organic tomato sauce. Frozen not included.",
            "item_info": '{"category_0":"Grocery","category_1":"Canned Goods","category_2":"Tomato Sauce"}',
            "sizing_comp": '{"size_user_friendly":"8 oz"}',
            "tags": "['organic']",
        }
    ]
    product = dataframe_to_products(rows)[0]
    assert "organic" in product.attribute_flags
    assert "tomato" in product.tokens_core
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalize.py::test_dataframe_to_products_extracts_attributes -v`
Expected: FAIL because attributes are missing

- [ ] **Step 3: Write minimal attribute extraction**

```python
# matcher/normalize.py
STOPWORDS = {"and", "the", "with", "for", "a", "an"}
ATTRIBUTE_KEYWORDS = {"organic", "frozen", "plain", "whole", "boneless", "gluten", "free"}


def extract_attribute_flags(text: str) -> list[str]:
    tokens = normalize_name(text).split()
    return sorted({token for token in tokens if token in ATTRIBUTE_KEYWORDS})


def extract_core_tokens(text: str) -> list[str]:
    tokens = normalize_name(text).split()
    return [token for token in tokens if token not in STOPWORDS and token not in ATTRIBUTE_KEYWORDS]
```

```python
# matcher/io.py
from matcher.normalize import (
    extract_attribute_flags,
    extract_core_tokens,
    extract_size,
    normalize_brand,
    normalize_name,
)


def dataframe_to_products(rows: list[dict[str, str]]) -> list[ProductRecord]:
    products = []
    for row in rows:
        combined_text = " ".join([row.get("name", ""), row.get("description", ""), row.get("tags", "")])
        sizing = parse_sizing_comp(row.get("sizing_comp", ""))
        size_value, size_unit = extract_size(sizing.get("size_user_friendly", "") or row.get("size_raw", ""))
        products.append(
            ProductRecord(
                item_id=row["item_id"],
                name=row["name"],
                brand_raw=row.get("brand_raw", ""),
                brand_norm=normalize_brand(row.get("brand_raw", "")),
                description=row.get("description", ""),
                category_path=parse_item_info(row.get("item_info", "")),
                size_value=size_value,
                size_unit=size_unit,
                tokens_core=extract_core_tokens(row["name"]),
                tokens_full=normalize_name(row["name"]).split(),
                attribute_flags=extract_attribute_flags(combined_text),
                raw_payload=row,
            )
        )
    return products
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/io.py matcher/normalize.py tests/test_normalize.py
git commit -m "feat: extract product attributes during normalization"
```

### Task 7: Build Broad Candidate Retrieval

**Files:**
- Create: `matcher/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing retrieval test**

```python
# tests/test_retrieval.py
from matcher.retrieval import retrieve_candidates
from matcher.schemas import ProductRecord


def test_retrieve_candidates_prefers_same_family():
    item_a = ProductRecord(item_id="a1", name="Organic Tomato Sauce", tokens_core=["tomato", "sauce"], category_path=["Grocery", "Sauces"])
    items_b = [
        ProductRecord(item_id="b1", name="Organic Tomato Sauce", tokens_core=["tomato", "sauce"], category_path=["Grocery", "Sauces"]),
        ProductRecord(item_id="b2", name="Dog Biscuits", tokens_core=["dog", "biscuits"], category_path=["Pets"]),
    ]
    candidates = retrieve_candidates(item_a, items_b, top_k=2)
    assert candidates[0].item_id == "b1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py -v`
Expected: FAIL with `ImportError` for `matcher.retrieval`

- [ ] **Step 3: Write minimal retrieval code**

```python
# matcher/retrieval.py
from rapidfuzz.fuzz import token_set_ratio


def _category_overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)


def retrieve_candidates(item_a, items_b, top_k: int = 200):
    scored = []
    query = " ".join(item_a.tokens_core)
    for item_b in items_b:
        text_score = token_set_ratio(query, " ".join(item_b.tokens_core)) / 100.0
        cat_score = _category_overlap(item_a.category_path, item_b.category_path)
        score = (0.7 * text_score) + (0.3 * cat_score)
        scored.append((score, item_b))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/retrieval.py tests/test_retrieval.py
git commit -m "feat: add broad candidate retrieval"
```

### Task 8: Expand Retrieval To Multi-Channel Search

**Files:**
- Modify: `matcher/retrieval.py`
- Modify: `matcher/config.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing multi-channel retrieval test**

```python
# tests/test_retrieval.py
from matcher.retrieval import build_retrieval_index, retrieve_candidates
from matcher.schemas import ProductRecord


def test_retrieval_uses_brand_and_size_signals():
    items_b = [
        ProductRecord(item_id="b1", name="Chobani Greek Yogurt Honey", brand_norm="chobani", tokens_core=["greek", "yogurt", "honey"], size_value=5.3, size_unit="oz"),
        ProductRecord(item_id="b2", name="Random Yogurt", brand_norm="other", tokens_core=["yogurt"], size_value=32.0, size_unit="oz"),
    ]
    item_a = ProductRecord(item_id="a1", name="Chobani Whole Milk Greek Yogurt Honey Blended 5.3 oz", brand_norm="chobani", tokens_core=["greek", "yogurt", "honey"], size_value=5.3, size_unit="oz")
    index = build_retrieval_index(items_b)
    candidates = retrieve_candidates(item_a, items_b, index=index, top_k=2)
    assert candidates[0].item_id == "b1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retrieval.py::test_retrieval_uses_brand_and_size_signals -v`
Expected: FAIL because `build_retrieval_index` is missing

- [ ] **Step 3: Write minimal index-based retrieval**

```python
# matcher/retrieval.py
from collections import defaultdict


def build_retrieval_index(items_b):
    by_brand = defaultdict(list)
    by_unit = defaultdict(list)
    for item in items_b:
        if item.brand_norm:
            by_brand[item.brand_norm].append(item)
        if item.size_unit:
            by_unit[item.size_unit].append(item)
    return {"by_brand": by_brand, "by_unit": by_unit}


def retrieve_candidates(item_a, items_b, index=None, top_k: int = 200):
    search_space = list(items_b)
    if index and item_a.brand_norm in index["by_brand"]:
        search_space.extend(index["by_brand"][item_a.brand_norm])
    if index and item_a.size_unit in index["by_unit"]:
        search_space.extend(index["by_unit"][item_a.size_unit])
    deduped = {item.item_id: item for item in search_space}.values()
    scored = []
    query = " ".join(item_a.tokens_core)
    for item_b in deduped:
        text_score = token_set_ratio(query, " ".join(item_b.tokens_core)) / 100.0
        cat_score = _category_overlap(item_a.category_path, item_b.category_path)
        brand_score = 1.0 if item_a.brand_norm and item_a.brand_norm == item_b.brand_norm else 0.0
        size_score = 1.0 if item_a.size_value and item_b.size_value and item_a.size_value == item_b.size_value and item_a.size_unit == item_b.size_unit else 0.0
        score = (0.5 * text_score) + (0.2 * cat_score) + (0.2 * brand_score) + (0.1 * size_score)
        scored.append((score, item_b))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/retrieval.py matcher/config.py tests/test_retrieval.py
git commit -m "feat: expand retrieval with brand and size channels"
```

### Task 9: Implement Deterministic Pair Scoring

**Files:**
- Create: `matcher/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing scoring tests**

```python
# tests/test_scoring.py
from matcher.scoring import score_candidate_pair
from matcher.schemas import ProductRecord


def test_score_candidate_pair_rewards_family_match():
    item_a = ProductRecord(item_id="a1", name="Organic Tomato Sauce", tokens_core=["tomato", "sauce"], category_path=["Grocery"], attribute_flags=["organic"])
    item_b = ProductRecord(item_id="b1", name="Wegmans Organic Tomato Sauce", tokens_core=["tomato", "sauce"], category_path=["Grocery"], attribute_flags=["organic"])
    score = score_candidate_pair(item_a, item_b)
    assert score.deterministic_score > 0.7


def test_score_candidate_pair_penalizes_hard_conflict():
    item_a = ProductRecord(item_id="a1", name="Dog Food", tokens_core=["dog", "food"], category_path=["Pets"])
    item_b = ProductRecord(item_id="b1", name="Tomato Sauce", tokens_core=["tomato", "sauce"], category_path=["Grocery"])
    score = score_candidate_pair(item_a, item_b)
    assert "category_conflict" in score.contradiction_flags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL with `ImportError` for `matcher.scoring`

- [ ] **Step 3: Write minimal deterministic scoring**

```python
# matcher/scoring.py
from matcher.schemas import CandidateScore


def score_candidate_pair(item_a, item_b) -> CandidateScore:
    token_overlap = len(set(item_a.tokens_core) & set(item_b.tokens_core))
    token_union = max(len(set(item_a.tokens_core) | set(item_b.tokens_core)), 1)
    token_score = token_overlap / token_union

    category_overlap = len(set(item_a.category_path) & set(item_b.category_path))
    category_union = max(len(set(item_a.category_path) | set(item_b.category_path)), 1)
    category_score = category_overlap / category_union if item_a.category_path or item_b.category_path else 0.0

    attribute_overlap = len(set(item_a.attribute_flags) & set(item_b.attribute_flags))
    attribute_union = max(len(set(item_a.attribute_flags) | set(item_b.attribute_flags)), 1)
    attribute_score = attribute_overlap / attribute_union if item_a.attribute_flags or item_b.attribute_flags else 0.0

    contradiction_flags = []
    if item_a.category_path and item_b.category_path and not set(item_a.category_path) & set(item_b.category_path):
        contradiction_flags.append("category_conflict")

    deterministic_score = (0.5 * token_score) + (0.3 * category_score) + (0.2 * attribute_score)
    if contradiction_flags:
        deterministic_score *= 0.25

    return CandidateScore(
        item_id_a=item_a.item_id,
        item_id_b=item_b.item_id,
        deterministic_score=deterministic_score,
        contradiction_flags=contradiction_flags,
        reason_codes=["token_overlap", "category_overlap"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/scoring.py tests/test_scoring.py
git commit -m "feat: add deterministic candidate scoring"
```

### Task 10: Add GPT-5 Nano Pairwise Scoring

**Files:**
- Create: `matcher/llm.py`
- Create: `matcher/persistence.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing LLM payload test**

```python
# tests/test_scoring.py
from matcher.llm import build_pair_prompt
from matcher.schemas import ProductRecord


def test_build_pair_prompt_includes_structured_fields():
    item_a = ProductRecord(item_id="a1", name="Organic Tomato Sauce", brand_norm="great value", category_path=["Grocery"], size_value=8.0, size_unit="oz")
    item_b = ProductRecord(item_id="b1", name="Wegmans Organic Tomato Sauce", brand_norm="wegmans", category_path=["Grocery"], size_value=8.0, size_unit="oz")
    prompt = build_pair_prompt(item_a, item_b)
    assert '"item_id":"a1"' in prompt
    assert '"item_id":"b1"' in prompt
    assert '"size_unit":"oz"' in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py::test_build_pair_prompt_includes_structured_fields -v`
Expected: FAIL with `ImportError` for `matcher.llm`

- [ ] **Step 3: Write minimal LLM and cache helpers**

```python
# matcher/persistence.py
from diskcache import Cache


def open_cache(path: str) -> Cache:
    return Cache(path)
```

```python
# matcher/llm.py
import json
from openai import OpenAI


SYSTEM_PROMPT = """You are a retail product matcher.
Return JSON with keys:
exact_match_score, substitute_match_score, confidence, reason_codes, contradictions.
"""


def build_pair_prompt(item_a, item_b) -> str:
    payload = {
        "a": {
            "item_id": item_a.item_id,
            "name": item_a.name,
            "brand_norm": item_a.brand_norm,
            "category_path": item_a.category_path,
            "size_value": item_a.size_value,
            "size_unit": item_a.size_unit,
            "attribute_flags": item_a.attribute_flags,
        },
        "b": {
            "item_id": item_b.item_id,
            "name": item_b.name,
            "brand_norm": item_b.brand_norm,
            "category_path": item_b.category_path,
            "size_value": item_b.size_value,
            "size_unit": item_b.size_unit,
            "attribute_flags": item_b.attribute_flags,
        },
    }
    return json.dumps(payload, separators=(",", ":"))


def build_openai_client() -> OpenAI:
    return OpenAI()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py::test_build_pair_prompt_includes_structured_fields -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/llm.py matcher/persistence.py tests/test_scoring.py
git commit -m "feat: add GPT pair prompt and cache helpers"
```

### Task 11: Parse Structured GPT Output And Blend Scores

**Files:**
- Modify: `matcher/llm.py`
- Modify: `matcher/scoring.py`
- Modify: `matcher/schemas.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing GPT parse test**

```python
# tests/test_scoring.py
from matcher.llm import parse_llm_response


def test_parse_llm_response_reads_confidence():
    raw = '{"exact_match_score":0.2,"substitute_match_score":0.9,"confidence":0.86,"reason_codes":["private_label_match"],"contradictions":[]}'
    parsed = parse_llm_response(raw)
    assert parsed["confidence"] == 0.86
    assert parsed["substitute_match_score"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py::test_parse_llm_response_reads_confidence -v`
Expected: FAIL because parser is missing

- [ ] **Step 3: Write minimal parse and blending code**

```python
# matcher/llm.py
import json


def parse_llm_response(raw: str) -> dict:
    data = json.loads(raw)
    return {
        "exact_match_score": float(data["exact_match_score"]),
        "substitute_match_score": float(data["substitute_match_score"]),
        "confidence": float(data["confidence"]),
        "reason_codes": list(data.get("reason_codes", [])),
        "contradictions": list(data.get("contradictions", [])),
    }
```

```python
# matcher/scoring.py
def blend_scores(deterministic_score: float, llm_confidence: float, substitute_match_score: float) -> float:
    return (0.3 * deterministic_score) + (0.4 * llm_confidence) + (0.3 * substitute_match_score)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/llm.py matcher/scoring.py tests/test_scoring.py
git commit -m "feat: parse GPT outputs and blend candidate scores"
```

### Task 12: Implement Final Match Resolution

**Files:**
- Create: `matcher/resolution.py`
- Test: `tests/test_resolution.py`

- [ ] **Step 1: Write the failing resolution test**

```python
# tests/test_resolution.py
from matcher.resolution import choose_best_match
from matcher.schemas import CandidateScore


def test_choose_best_match_marks_low_quality_for_weak_best_guess():
    candidates = [
        CandidateScore(item_id_a="a1", item_id_b="b1", deterministic_score=0.2, llm_score=0.28, combined_score=0.27),
        CandidateScore(item_id_a="a1", item_id_b="b2", deterministic_score=0.1, llm_score=0.14, combined_score=0.13),
    ]
    decision = choose_best_match(candidates)
    assert decision.item_id_b == "b1"
    assert decision.match_quality == "low"
    assert decision.review_flag is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution.py -v`
Expected: FAIL with `ImportError` for `matcher.resolution`

- [ ] **Step 3: Write minimal resolution logic**

```python
# matcher/resolution.py
from matcher.schemas import MatchDecision


def choose_best_match(candidates):
    best = sorted(candidates, key=lambda item: item.combined_score or 0.0, reverse=True)[0]
    confidence = best.combined_score or 0.0
    if confidence >= 0.85:
        match_quality = "high"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    elif confidence >= 0.55:
        match_quality = "medium"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    else:
        match_quality = "low"
        decision_source = "fallback"
        review_flag = True
    return MatchDecision(
        item_id_a=best.item_id_a,
        item_id_b=best.item_id_b,
        confidence=confidence,
        match_quality=match_quality,
        decision_source=decision_source,
        review_flag=review_flag,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/resolution.py tests/test_resolution.py
git commit -m "feat: add final match resolution"
```

### Task 13: Build End-To-End Pipeline Orchestration

**Files:**
- Create: `matcher/pipeline.py`
- Modify: `matcher/io.py`
- Modify: `matcher/retrieval.py`
- Modify: `matcher/scoring.py`
- Modify: `matcher/llm.py`
- Modify: `matcher/resolution.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_pipeline_smoke.py
from matcher.pipeline import run_pipeline


def test_run_pipeline_returns_one_match_per_input_row(tmp_path):
    rows_a = [
        {"item_id": "a1", "name": "Organic Tomato Sauce 8 oz", "brand_raw": "Great Value", "description": "", "item_info": '{"category_0":"Grocery","category_1":"Sauces"}', "sizing_comp": '{"size_user_friendly":"8 oz"}', "tags": "[]"},
        {"item_id": "a2", "name": "Dog Food", "brand_raw": "Pedigree", "description": "", "item_info": '{"category_0":"Pets"}', "sizing_comp": '{}', "tags": "[]"},
    ]
    rows_b = [
        {"item_id": "b1", "name": "Wegmans Organic Tomato Sauce 8 oz", "brand_raw": "Wegmans", "description": "", "item_info": '{"category_0":"Grocery","category_1":"Sauces"}', "sizing_comp": '{"size_user_friendly":"8 oz"}', "tags": "[]"},
        {"item_id": "b2", "name": "Pedigree Dog Food", "brand_raw": "Pedigree", "description": "", "item_info": '{"category_0":"Pets"}', "sizing_comp": '{}', "tags": "[]"},
    ]
    matches = run_pipeline(rows_a, rows_b, llm_enabled=False)
    assert len(matches) == 2
    assert {match.item_id_a for match in matches} == {"a1", "a2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_smoke.py::test_run_pipeline_returns_one_match_per_input_row -v`
Expected: FAIL with `ImportError` for `matcher.pipeline`

- [ ] **Step 3: Write minimal pipeline orchestration**

```python
# matcher/pipeline.py
from matcher.io import dataframe_to_products
from matcher.retrieval import build_retrieval_index, retrieve_candidates
from matcher.scoring import score_candidate_pair, blend_scores
from matcher.resolution import choose_best_match


def run_pipeline(rows_a, rows_b, llm_enabled: bool = False):
    products_a = dataframe_to_products(rows_a)
    products_b = dataframe_to_products(rows_b)
    index = build_retrieval_index(products_b)
    decisions = []
    for item_a in products_a:
        candidates = retrieve_candidates(item_a, products_b, index=index, top_k=25)
        scored = []
        for item_b in candidates:
            pair = score_candidate_pair(item_a, item_b)
            pair.combined_score = blend_scores(pair.deterministic_score, pair.llm_score or 0.0, pair.llm_score or 0.0)
            scored.append(pair)
        decisions.append(choose_best_match(scored))
    return decisions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_smoke.py::test_run_pipeline_returns_one_match_per_input_row -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/pipeline.py matcher/io.py matcher/retrieval.py matcher/scoring.py matcher/resolution.py tests/test_pipeline_smoke.py
git commit -m "feat: add end-to-end matching pipeline"
```

### Task 14: Add CLI, Outputs, And Summary Reporting

**Files:**
- Create: `matcher/cli.py`
- Modify: `matcher/io.py`
- Modify: `matcher/persistence.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing CLI output test**

```python
# tests/test_pipeline_smoke.py
from pathlib import Path
from matcher.io import write_matches_csv
from matcher.schemas import MatchDecision


def test_write_matches_csv_outputs_expected_columns(tmp_path: Path):
    path = tmp_path / "matches.csv"
    decisions = [
        MatchDecision(item_id_a="a1", item_id_b="b1", confidence=0.92, match_quality="high", decision_source="llm", review_flag=False)
    ]
    write_matches_csv(path, decisions)
    content = path.read_text()
    assert "item_id_a,item_id_b,confidence,match_quality,decision_source,review_flag" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_smoke.py::test_write_matches_csv_outputs_expected_columns -v`
Expected: FAIL because `write_matches_csv` is missing

- [ ] **Step 3: Write minimal output and CLI code**

```python
# matcher/io.py
import csv
from pathlib import Path


def write_matches_csv(path: Path, decisions):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["item_id_a", "item_id_b", "confidence", "match_quality", "decision_source", "review_flag"],
        )
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.model_dump())
```

```python
# matcher/cli.py
from pathlib import Path
from matcher.io import load_catalog_csv, write_matches_csv


def main():
    input_a = load_catalog_csv("grocery_store_a_items_final.csv").to_dict("records")
    input_b = load_catalog_csv("grocery_store_b_items_final.csv").to_dict("records")
    from matcher.pipeline import run_pipeline

    decisions = run_pipeline(input_a, input_b, llm_enabled=True)
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matches_csv(output_dir / "matches.csv", decisions)
    print(f"matches={len(decisions)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_smoke.py::test_write_matches_csv_outputs_expected_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/cli.py matcher/io.py tests/test_pipeline_smoke.py
git commit -m "feat: add CLI and CSV output writer"
```

### Task 15: Add GPT Execution Path And Structured Logs

**Files:**
- Modify: `matcher/llm.py`
- Modify: `matcher/persistence.py`
- Modify: `matcher/pipeline.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing log test**

```python
# tests/test_pipeline_smoke.py
from pathlib import Path
from matcher.persistence import append_match_log


def test_append_match_log_writes_jsonl(tmp_path: Path):
    path = tmp_path / "match_logs.jsonl"
    append_match_log(path, {"item_id_a": "a1", "item_id_b": "b1", "combined_score": 0.91})
    content = path.read_text().strip()
    assert '"item_id_a":"a1"' in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_smoke.py::test_append_match_log_writes_jsonl -v`
Expected: FAIL because `append_match_log` is missing

- [ ] **Step 3: Write minimal GPT execution and logging code**

```python
# matcher/persistence.py
import orjson


def append_match_log(path, payload: dict):
    with open(path, "ab") as handle:
        handle.write(orjson.dumps(payload))
        handle.write(b"\n")
```

```python
# matcher/llm.py
def score_pair_with_llm(client, model: str, item_a, item_b) -> dict:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_pair_prompt(item_a, item_b)},
        ],
    )
    return parse_llm_response(response.output_text)
```

```python
# matcher/pipeline.py
from matcher.llm import build_openai_client, score_pair_with_llm
from matcher.persistence import append_match_log


def run_pipeline(rows_a, rows_b, llm_enabled: bool = False, output_dir: str = "artifacts"):
    products_a = dataframe_to_products(rows_a)
    products_b = dataframe_to_products(rows_b)
    index = build_retrieval_index(products_b)
    client = build_openai_client() if llm_enabled else None
    decisions = []
    for item_a in products_a:
        candidates = retrieve_candidates(item_a, products_b, index=index, top_k=25)
        scored = []
        for item_b in candidates:
            pair = score_candidate_pair(item_a, item_b)
            if llm_enabled:
                llm_result = score_pair_with_llm(client, "gpt-5-nano", item_a, item_b)
                pair.llm_score = llm_result["confidence"]
                pair.reason_codes.extend(llm_result["reason_codes"])
                pair.contradiction_flags.extend(llm_result["contradictions"])
                pair.combined_score = blend_scores(pair.deterministic_score, llm_result["confidence"], llm_result["substitute_match_score"])
            else:
                pair.combined_score = blend_scores(pair.deterministic_score, 0.0, 0.0)
            scored.append(pair)
        decision = choose_best_match(scored)
        append_match_log(
            f"{output_dir}/match_logs.jsonl",
            {
                "item_id_a": decision.item_id_a,
                "item_id_b": decision.item_id_b,
                "confidence": decision.confidence,
                "match_quality": decision.match_quality,
                "decision_source": decision.decision_source,
                "review_flag": decision.review_flag,
            },
        )
        decisions.append(decision)
    return decisions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_smoke.py::test_append_match_log_writes_jsonl -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/llm.py matcher/persistence.py matcher/pipeline.py tests/test_pipeline_smoke.py
git commit -m "feat: add GPT scoring path and JSONL logs"
```

### Task 16: Add Pair Cache And Retry Logic

**Files:**
- Modify: `matcher/llm.py`
- Modify: `matcher/persistence.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing cache test**

```python
# tests/test_scoring.py
from matcher.persistence import open_cache


def test_cache_round_trip(tmp_path):
    cache = open_cache(str(tmp_path / "cache"))
    cache["pair:a1:b1"] = {"confidence": 0.8}
    assert cache["pair:a1:b1"]["confidence"] == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring.py::test_cache_round_trip -v`
Expected: FAIL if cache helper does not behave as expected

- [ ] **Step 3: Write minimal cache-backed LLM wrapper**

```python
# matcher/llm.py
from tenacity import retry, stop_after_attempt, wait_exponential


def pair_cache_key(item_a, item_b) -> str:
    return f"pair:{item_a.item_id}:{item_b.item_id}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def score_pair_with_llm_cached(client, cache, model: str, item_a, item_b) -> dict:
    key = pair_cache_key(item_a, item_b)
    if key in cache:
        return cache[key]
    result = score_pair_with_llm(client, model, item_a, item_b)
    cache[key] = result
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scoring.py::test_cache_round_trip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/llm.py matcher/persistence.py tests/test_scoring.py
git commit -m "feat: cache GPT pair scores with retries"
```

### Task 17: Tune Retrieval Width, LLM Width, And Quality Labels

**Files:**
- Modify: `matcher/config.py`
- Modify: `matcher/pipeline.py`
- Modify: `matcher/resolution.py`
- Test: `tests/test_resolution.py`

- [ ] **Step 1: Write the failing threshold test**

```python
# tests/test_resolution.py
from matcher.resolution import choose_best_match
from matcher.schemas import CandidateScore


def test_choose_best_match_marks_medium_range_correctly():
    candidates = [
        CandidateScore(item_id_a="a1", item_id_b="b1", deterministic_score=0.4, llm_score=0.7, combined_score=0.64),
        CandidateScore(item_id_a="a1", item_id_b="b2", deterministic_score=0.35, llm_score=0.55, combined_score=0.51),
    ]
    decision = choose_best_match(candidates)
    assert decision.match_quality == "medium"
    assert decision.decision_source == "llm"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_resolution.py::test_choose_best_match_marks_medium_range_correctly -v`
Expected: FAIL if thresholds are off

- [ ] **Step 3: Write minimal threshold tuning**

```python
# matcher/config.py
class Settings(BaseModel):
    input_a_path: Path = Path("grocery_store_a_items_final.csv")
    input_b_path: Path = Path("grocery_store_b_items_final.csv")
    output_dir: Path = Path("artifacts")
    cache_dir: Path = Path("artifacts/cache")
    llm_model: str = "gpt-5-nano"
    retrieval_k: int = 200
    llm_top_n: int = 24
    high_quality_threshold: float = 0.85
    medium_quality_threshold: float = 0.55
```

```python
# matcher/resolution.py
def choose_best_match(candidates, high_quality_threshold: float = 0.85, medium_quality_threshold: float = 0.55):
    best = sorted(candidates, key=lambda item: item.combined_score or 0.0, reverse=True)[0]
    confidence = best.combined_score or 0.0
    if confidence >= high_quality_threshold:
        match_quality = "high"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    elif confidence >= medium_quality_threshold:
        match_quality = "medium"
        decision_source = "llm" if best.llm_score is not None else "rules"
        review_flag = False
    else:
        match_quality = "low"
        decision_source = "fallback"
        review_flag = True
    return MatchDecision(
        item_id_a=best.item_id_a,
        item_id_b=best.item_id_b,
        confidence=confidence,
        match_quality=match_quality,
        decision_source=decision_source,
        review_flag=review_flag,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_resolution.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/config.py matcher/resolution.py tests/test_resolution.py
git commit -m "feat: tune quality thresholds for match labels"
```

### Task 18: Run Full Verification And First Real Execution

**Files:**
- Modify: `matcher/cli.py`
- Modify: `matcher/pipeline.py`
- Test: `tests/test_parsing.py`
- Test: `tests/test_normalize.py`
- Test: `tests/test_retrieval.py`
- Test: `tests/test_scoring.py`
- Test: `tests/test_resolution.py`
- Test: `tests/test_pipeline_smoke.py`

- [ ] **Step 1: Write the failing run-summary test**

```python
# tests/test_pipeline_smoke.py
from matcher.pipeline import summarize_decisions
from matcher.schemas import MatchDecision


def test_summarize_decisions_counts_quality_bands():
    decisions = [
        MatchDecision(item_id_a="a1", item_id_b="b1", confidence=0.91, match_quality="high", decision_source="llm", review_flag=False),
        MatchDecision(item_id_a="a2", item_id_b="b2", confidence=0.42, match_quality="low", decision_source="fallback", review_flag=True),
    ]
    summary = summarize_decisions(decisions)
    assert summary["total_matches"] == 2
    assert summary["by_quality"]["high"] == 1
    assert summary["by_quality"]["low"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_smoke.py::test_summarize_decisions_counts_quality_bands -v`
Expected: FAIL because `summarize_decisions` is missing

- [ ] **Step 3: Write minimal summary code and run full verification**

```python
# matcher/pipeline.py
from collections import Counter


def summarize_decisions(decisions):
    quality_counts = Counter(decision.match_quality for decision in decisions)
    source_counts = Counter(decision.decision_source for decision in decisions)
    return {
        "total_matches": len(decisions),
        "by_quality": dict(quality_counts),
        "by_source": dict(source_counts),
    }
```

```python
# matcher/cli.py
from matcher.pipeline import run_pipeline, summarize_decisions


def main():
    input_a = load_catalog_csv("grocery_store_a_items_final.csv").to_dict("records")
    input_b = load_catalog_csv("grocery_store_b_items_final.csv").to_dict("records")
    decisions = run_pipeline(input_a, input_b, llm_enabled=True)
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_matches_csv(output_dir / "matches.csv", decisions)
    summary = summarize_decisions(decisions)
    print(summary)
```

Run: `pytest -v`
Expected: PASS for all tests

Run: `python -m matcher.cli`
Expected: writes `artifacts/matches.csv`, writes `artifacts/match_logs.jsonl`, prints summary dictionary with `total_matches`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_smoke.py::test_summarize_decisions_counts_quality_bands -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add matcher/cli.py matcher/pipeline.py tests/test_pipeline_smoke.py
git commit -m "feat: add run summary and first full execution path"
```

## Self-Review

Spec coverage check:

- normalization and feature extraction: Tasks 3, 4, 5, 6
- broad retrieval over all B rows: Tasks 7, 8
- deterministic scoring: Task 9
- GPT-heavy semantic scoring: Tasks 10, 11, 15, 16
- final one-best-match resolution: Tasks 12, 17
- persistence and reusable caches: Tasks 10, 15, 16
- output CSV, logs, summary: Tasks 14, 15, 18

Placeholder scan:

- no `TODO`, `TBD`, or deferred implementation markers were left in the steps

Type consistency:

- `ProductRecord`, `CandidateScore`, and `MatchDecision` are introduced before later tasks use them
- `run_pipeline`, `choose_best_match`, `build_pair_prompt`, and `summarize_decisions` names stay consistent across tasks
