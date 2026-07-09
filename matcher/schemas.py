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
