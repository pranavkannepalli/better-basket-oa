import json
import os

from dotenv import load_dotenv
from openai import BadRequestError
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from matcher.config import Settings

load_dotenv()
_settings = Settings()


SYSTEM_PROMPT = """You are a retail product matcher for grocery price indexing.
Your job is to choose the single closest product from store B for one product from store A.

Important rules:
- Prefer customer-equivalent matches, not literal string matches.
- Use product name, description, category, size, form, and attribute flags heavily.
- Do not require exact brand match for private label, fresh, or loose products.
- Exact matches should usually be national-brand same-item matches.
- Non-exact matches are valid when a normal shopper would consider the two products essentially the same thing.
- If names differ but description, size, form, and category mostly line up, that can still be a strong match.
- Only choose from the provided shortlist.

Scoring guidance:
- Return numeric confidence values from 0.0 to 1.0.
- High confidence: mostly the same product for a shopper, even if private-label brand/name differs.
- Medium confidence: probably the right substitute/match, but some uncertainty remains.
- Low confidence: weak evidence or major mismatch.
"""


class ShortlistLLMResponse(BaseModel):
    chosen_item_id_b: str
    exact_match_score: float = Field(ge=0.0, le=1.0)
    substitute_match_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: list[str]
    contradictions: list[str]


def _serialize_product(item) -> dict:
    return {
        "item_id": item.item_id,
        "name": item.name,
        "description": item.description,
        "private_label_flag": item.private_label_flag,
        "brand_norm": item.brand_norm,
        "category_path": item.category_path,
        "size_value": item.size_value,
        "size_unit": item.size_unit,
        "pack_count": item.pack_count,
        "form_flags": item.form_flags,
        "attribute_flags": item.attribute_flags,
    }


def build_shortlist_prompt(item_a, candidates) -> str:
    payload = {
        "a": _serialize_product(item_a),
        "candidates_b": [_serialize_product(item_b) for item_b in candidates],
    }
    return json.dumps(payload, separators=(",", ":"))


def build_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def fallback_shortlist_result(candidates, reason_code: str) -> dict:
    chosen_item_id_b = candidates[0].item_id if candidates else ""
    return {
        "chosen_item_id_b": chosen_item_id_b,
        "exact_match_score": 0.0,
        "substitute_match_score": 0.0,
        "confidence": 0.0,
        "reason_codes": [reason_code],
        "contradictions": [],
    }


def shortlist_cache_key(item_a, candidates) -> str:
    candidate_ids = ",".join(item.item_id for item in candidates)
    return f"shortlist:{item_a.item_id}:{candidate_ids}"


def _is_content_filter_error(exc: BadRequestError) -> bool:
    body = getattr(exc, "body", {}) or {}
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if error.get("code") == "content_filter":
        return True
    response = getattr(exc, "response", None)
    response_text = getattr(response, "text", "") if response is not None else ""
    return "content_filter" in str(exc).lower() or "content_filter" in str(response_text).lower()


def _normalize_score(value) -> float:
    if isinstance(value, str):
        band = value.strip().lower()
        if band == "high":
            return 0.85
        if band == "medium":
            return 0.58
        if band == "low":
            return 0.20
    numeric = float(value)
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    return max(0.0, min(1.0, numeric))


def sanitize_llm_result(result: dict, candidates) -> dict:
    allowed_ids = {candidate.item_id for candidate in candidates}
    chosen_item_id_b = str(result.get("chosen_item_id_b", ""))
    reason_codes = list(result.get("reason_codes", []))
    if chosen_item_id_b not in allowed_ids:
        chosen_item_id_b = candidates[0].item_id if candidates else ""
        reason_codes.append("invalid_choice_fallback")
    return {
        "chosen_item_id_b": chosen_item_id_b,
        "exact_match_score": _normalize_score(result["exact_match_score"]),
        "substitute_match_score": _normalize_score(result["substitute_match_score"]),
        "confidence": _normalize_score(result["confidence"]),
        "reason_codes": reason_codes,
        "contradictions": list(result.get("contradictions", [])),
    }


def parse_llm_response(raw: str, candidates=None) -> dict:
    data = json.loads(raw)
    parsed = {
        "chosen_item_id_b": str(data.get("chosen_item_id_b", "")),
        "exact_match_score": _normalize_score(data["exact_match_score"]),
        "substitute_match_score": _normalize_score(data["substitute_match_score"]),
        "confidence": _normalize_score(data["confidence"]),
        "reason_codes": list(data.get("reason_codes", [])),
        "contradictions": list(data.get("contradictions", [])),
    }
    if candidates is not None:
        return sanitize_llm_result(parsed, candidates)
    return parsed


def score_shortlist_with_llm(client, model: str, item_a, candidates) -> dict:
    response = client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=build_shortlist_prompt(item_a, candidates),
        text_format=ShortlistLLMResponse,
        temperature=_settings.llm_temperature,
    )
    return sanitize_llm_result(response.output_parsed.model_dump(), candidates)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def score_shortlist_with_llm_cached(client, cache, model: str, item_a, candidates) -> dict:
    key = shortlist_cache_key(item_a, candidates)
    if key in cache:
        return cache[key]
    try:
        result = score_shortlist_with_llm(client, model, item_a, candidates)
    except BadRequestError as exc:
        if _is_content_filter_error(exc):
            result = fallback_shortlist_result(candidates, "content_filter_fallback")
        else:
            raise
    cache[key] = result
    return result
