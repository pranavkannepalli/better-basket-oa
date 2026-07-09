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
