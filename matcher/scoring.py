from matcher.schemas import CandidateScore
from matcher.product_rules import cheese_variety_conflict
from matcher.product_rules import diet_variant_conflict
from matcher.product_rules import product_type_groups
from matcher.product_rules import product_types_compatible
from rapidfuzz.fuzz import token_set_ratio

HARD_CONTRADICTIONS = {
    "category_conflict",
    "diet_variant_conflict",
    "form_conflict",
    "pack_count_conflict",
    "product_type_conflict",
    "size_conflict",
    "variety_conflict",
}
GROCERY_LIKE_CATEGORIES = {"food", "grocery", "frozen", "produce", "dairy", "meat", "seafood", "bakery"}


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def blend_scores(deterministic_score: float, llm_confidence: float, substitute_match_score: float) -> float:
    blended = (
        (0.2 * _clamp_score(deterministic_score))
        + (0.5 * _clamp_score(llm_confidence))
        + (0.3 * _clamp_score(substitute_match_score))
    )
    return _clamp_score(blended)


def deterministic_only_score(deterministic_score: float) -> float:
    return _clamp_score(deterministic_score)


def _overlap_score(values_a: set[str], values_b: set[str]) -> float:
    if not values_a and not values_b:
        return 0.0
    return len(values_a & values_b) / max(len(values_a | values_b), 1)


def _normalize_category(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in GROCERY_LIKE_CATEGORIES:
        return "grocery"
    return normalized


def _category_set(path: list[str]) -> set[str]:
    return {_normalize_category(value) for value in path if value}


def _top_categories_compatible(path_a: list[str], path_b: list[str]) -> bool:
    if not path_a or not path_b:
        return True
    return _normalize_category(path_a[0]) == _normalize_category(path_b[0])


def _size_score(item_a, item_b) -> tuple[float, list[str]]:
    if item_a.size_value is None or item_b.size_value is None:
        return 0.0, []
    if item_a.size_unit != item_b.size_unit:
        return 0.0, ["size_unit_conflict"]
    larger = max(item_a.size_value, item_b.size_value)
    if larger == 0:
        return 0.0, []
    ratio_delta = abs(item_a.size_value - item_b.size_value) / larger
    if ratio_delta <= 0.05:
        return 1.0, []
    if ratio_delta <= 0.25:
        return 0.4, []
    return 0.0, ["size_conflict"]


def _pack_score(item_a, item_b) -> tuple[float, list[str]]:
    if item_a.pack_count is None or item_b.pack_count is None:
        return 0.0, []
    if item_a.pack_count == item_b.pack_count:
        return 1.0, []
    return 0.0, ["pack_count_conflict"]


def _brand_score(item_a, item_b) -> float:
    if not item_a.brand_norm or not item_b.brand_norm:
        return 0.0
    if item_a.brand_norm == item_b.brand_norm:
        return 1.0
    if item_a.private_label_flag and item_b.private_label_flag:
        return 0.7
    return 0.0


def score_candidate_pair(item_a, item_b) -> CandidateScore:
    token_a = set(item_a.tokens_core)
    token_b = set(item_b.tokens_core)
    category_a = _category_set(item_a.category_path)
    category_b = _category_set(item_b.category_path)
    attribute_a = set(item_a.attribute_flags)
    attribute_b = set(item_b.attribute_flags)

    form_a = set(item_a.form_flags)
    form_b = set(item_b.form_flags)
    product_types_a = product_type_groups(token_a, item_a.category_path)
    product_types_b = product_type_groups(token_b, item_b.category_path)

    token_score = max(
        _overlap_score(token_a, token_b),
        token_set_ratio(" ".join(item_a.tokens_core), " ".join(item_b.tokens_core)) / 100.0,
    )
    name_score = token_set_ratio(item_a.name, item_b.name) / 100.0
    category_score = _overlap_score(category_a, category_b)
    attribute_score = _overlap_score(attribute_a, attribute_b)
    form_score = _overlap_score(form_a, form_b)
    brand_score = _brand_score(item_a, item_b)
    size_score, size_flags = _size_score(item_a, item_b)
    pack_score, pack_flags = _pack_score(item_a, item_b)

    contradiction_flags = []
    if not _top_categories_compatible(item_a.category_path, item_b.category_path):
        contradiction_flags.append("category_conflict")
    if form_a and form_b and not (form_a & form_b):
        contradiction_flags.append("form_conflict")
    if not product_types_compatible(product_types_a, product_types_b):
        contradiction_flags.append("product_type_conflict")
    normalized_tokens_a = {token.lower() for token in token_a}
    normalized_tokens_b = {token.lower() for token in token_b}
    if diet_variant_conflict(normalized_tokens_a, normalized_tokens_b):
        contradiction_flags.append("diet_variant_conflict")
    if cheese_variety_conflict(normalized_tokens_a, normalized_tokens_b):
        contradiction_flags.append("variety_conflict")
    contradiction_flags.extend(size_flags)
    contradiction_flags.extend(pack_flags)

    deterministic_score = (
        (0.35 * token_score)
        + (0.15 * name_score)
        + (0.15 * category_score)
        + (0.10 * attribute_score)
        + (0.10 * size_score)
        + (0.10 * brand_score)
        + (0.025 * pack_score)
        + (0.025 * form_score)
    )
    if contradiction_flags:
        deterministic_score *= 0.25
    deterministic_score = _clamp_score(deterministic_score)

    return CandidateScore(
        item_id_a=item_a.item_id,
        item_id_b=item_b.item_id,
        deterministic_score=deterministic_score,
        contradiction_flags=contradiction_flags,
        reason_codes=["token_overlap", "category_overlap", "size_match", "brand_match"],
    )
