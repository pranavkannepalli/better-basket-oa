import re


UNIT_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fl\.?\s*oz|oz|lb|ct)")
STOPWORDS = {"and", "the", "with", "for", "a", "an"}
ATTRIBUTE_KEYWORDS = {"organic", "frozen", "plain", "whole", "boneless", "gluten", "free"}


def normalize_brand(value: str) -> str:
    return " ".join(value.lower().strip().split())


def normalize_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9\.\s]", " ", value)
    return " ".join(value.split())


def extract_attribute_flags(text: str) -> list[str]:
    tokens = normalize_name(text).split()
    return sorted({token for token in tokens if token in ATTRIBUTE_KEYWORDS})


def extract_core_tokens(text: str) -> list[str]:
    tokens = normalize_name(text).split()
    return [token for token in tokens if token not in STOPWORDS and token not in ATTRIBUTE_KEYWORDS]


def extract_size(value: str) -> tuple[float | None, str]:
    match = UNIT_PATTERN.search(value.lower())
    if not match:
        return None, ""
    unit = match.group("unit").replace(".", "")
    unit = " ".join(unit.split())
    return float(match.group("value")), unit
