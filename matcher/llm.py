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


def parse_llm_response(raw: str) -> dict:
    data = json.loads(raw)
    return {
        "exact_match_score": float(data["exact_match_score"]),
        "substitute_match_score": float(data["substitute_match_score"]),
        "confidence": float(data["confidence"]),
        "reason_codes": list(data.get("reason_codes", [])),
        "contradictions": list(data.get("contradictions", [])),
    }
