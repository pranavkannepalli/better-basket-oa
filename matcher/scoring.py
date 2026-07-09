from matcher.schemas import CandidateScore


def score_candidate_pair(item_a, item_b) -> CandidateScore:
    token_a = set(item_a.tokens_core)
    token_b = set(item_b.tokens_core)
    category_a = set(item_a.category_path)
    category_b = set(item_b.category_path)
    attribute_a = set(item_a.attribute_flags)
    attribute_b = set(item_b.attribute_flags)

    token_union = max(len(token_a | token_b), 1)
    token_score = len(token_a & token_b) / token_union

    if item_a.category_path or item_b.category_path:
        category_union = max(len(category_a | category_b), 1)
        category_score = len(category_a & category_b) / category_union
    else:
        category_score = 0.0

    if item_a.attribute_flags or item_b.attribute_flags:
        attribute_union = max(len(attribute_a | attribute_b), 1)
        attribute_score = len(attribute_a & attribute_b) / attribute_union
    else:
        attribute_score = 0.0

    contradiction_flags = []
    if item_a.category_path and item_b.category_path and not (category_a & category_b):
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
