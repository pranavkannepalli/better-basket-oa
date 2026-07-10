import re


UNIT_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fl\.?\s*oz|ounce|ounces|oz|pound|pounds|lb|lbs|count|ct)")
PACK_COUNT_PATTERN = re.compile(r"(?P<count>\d+)\s*(?:count|ct|pack|pk|ea|each)\b")
STOPWORDS = {"and", "the", "with", "for", "a", "an"}
ATTRIBUTE_KEYWORDS = {
    "boneless",
    "free",
    "fragrance",
    "gluten",
    "grass",
    "hypoallergenic",
    "low",
    "organic",
    "plain",
    "reduced",
    "sodium",
    "sugar",
    "unscented",
    "whole",
}
FORM_KEYWORDS = {"dry", "fresh", "frozen", "refrigerated"}


def normalize_brand(value: str) -> str:
    return " ".join(value.lower().strip().split())


def normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\.\s]", " ", value)
    return " ".join(value.split())


def extract_attribute_flags(text: str) -> list[str]:
    tokens = normalize_name(text).split()
    return sorted({token for token in tokens if token in ATTRIBUTE_KEYWORDS})


def extract_form_flags(text: str) -> list[str]:
    tokens = normalize_name(text).split()
    return sorted({token for token in tokens if token in FORM_KEYWORDS})


def extract_core_tokens(text: str) -> list[str]:
    tokens = normalize_name(text).split()
    return [
        token
        for token in tokens
        if token not in STOPWORDS and token not in ATTRIBUTE_KEYWORDS and token not in FORM_KEYWORDS
    ]


def extract_pack_count(text: str) -> int | None:
    match = PACK_COUNT_PATTERN.search(normalize_name(text))
    if not match:
        return None
    return int(match.group("count"))


def extract_size(value: str) -> tuple[float | None, str]:
    match = UNIT_PATTERN.search(value.lower())
    if not match:
        return None, ""
    unit = match.group("unit").replace(".", "")
    unit = " ".join(unit.split())
    unit = {
        "ounce": "oz",
        "ounces": "oz",
        "pound": "lb",
        "pounds": "lb",
        "lbs": "lb",
        "count": "ct",
    }.get(unit, unit)
    return float(match.group("value")), unit
